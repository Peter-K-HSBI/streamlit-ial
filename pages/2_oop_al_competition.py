from dataclasses import replace
import numpy as np
from sklearn.datasets import load_iris, make_moons
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from scipy.special import entr
from sklearn.metrics import accuracy_score
import operator
import datetime
from pathlib import Path

def PlotContour2(x_grid, y_grid, clf):
    # This function generates a grid of function values according to the classifier.
    zz = np.zeros((y_grid.size,x_grid.size))
    for i, x in enumerate(x_grid):
        grid_array = [(x,yval) for yval in y_grid]
        zz[:,i] = clf.predict(grid_array)
    return zz

def PlotUncert2(x_grid,y_grid,clf,unc):
    zz = np.zeros((y_grid.size,x_grid.size))
    for i, x in enumerate(x_grid):
        grid_array = [(x,yval) for yval in y_grid]
        predicted_probs = clf.predict_proba(grid_array)
        if unc == "entropy":
            zz[:,i] = entr(predicted_probs).sum(axis=1)
        elif unc == "lconf":
            zz[:,i] = 1-np.max(predicted_probs,axis=1)
        elif unc == "marg":
            if predicted_probs.shape[1]>1:
                sorted_probs = np.sort(predicted_probs)
                zz[:,i] = sorted_probs[:,-1]-sorted_probs[:,-2]
        else:
            print("Uncertainty function unknown, defaulting to entropy")
            zz[:,i] = entr(predicted_probs).sum(axis=1)
    return zz

def onpick(event, game): #the following will be what happens once a point is clicked.
    global t
    # The part about datetime is a workaround for the following problem:
    # When clicking at a place that is occupied by two points, the picker triggers two events.
    # This can happen accidentially when clicking on a point which is close to another one.
    # To prevent this from selecting two points, I check if the previous point has been selected in less than 1.4 seconds.
    # When testing, the double triggers always had a time difference below 1s, so this should do the trick.
    # Update: 1.1s was too close, there were still double triggers.
    t2=datetime.datetime.now()
    if t2-t < datetime.timedelta(seconds=1.4):
       return None
    t = datetime.datetime.now()
    game.N_QUERIES += 1
    artist = event.artist
    artist.set_picker(None)
    #artist.set_edgecolor("Black")
    selected_point = artist.get_offsets()
    query_index = np.where(game.X_pool == selected_point[0])[0][0]
    raw_index = np.where(game.X_raw == selected_point[0])[0][0]
    game.training_indices.append(raw_index.item())
    X, y = game.X_pool[query_index].reshape(1, -1), game.y_pool[query_index].reshape(1, )
    #add the new labeled point to the training data
    game.X_train = np.concatenate((game.X_train,X),axis=0)
    game.y_train = np.append(game.y_train,y)
    # Remove the queried instance from the unlabeled pool.
    game.X_pool, game.y_pool = np.delete(game.X_pool, query_index, axis=0), np.delete(game.y_pool, query_index)
    #artist.set_facecolor(props[y[0]])

    # AL choice: #
    game.clf.fit(game.X_al_train, game.y_al_train)
    al_query_index = np.argmax(entr(game.clf.predict_proba(game.X_al_pool)).sum(axis=1))
    X, y = game.X_al_pool[al_query_index].reshape(1, -1), game.y_al_pool[al_query_index].reshape(1, )
    al_raw_index = np.where(game.X_raw == X)[0][0]
    #add the new labeled point to the AL training data
    game.X_al_train = np.concatenate((game.X_al_train,X),axis=0)
    game.y_al_train = np.append(game.y_al_train,y)
    game.al_training_indices.append(al_raw_index.item())
    # Remove the queried instance from the unlabeled pool.
    game.X_al_pool, game.y_al_pool = np.delete(game.X_al_pool, al_query_index, axis=0), np.delete(game.y_al_pool, al_query_index)
    game.update_model()
    
# Function to toggle visibility of decision boundaries and uncertainty overlays.
def callback(label, game):
    ln = game.overlay_by_label[label]
    ln.set_visible(not ln.get_visible())
    game.fig.canvas.draw_idle()

# Function for uncertainty recalculation.
def callbutton(event,game):
    if len(game.X_train)!=0:
        game.clf.fit(game.X_train, game.y_train)
        min1, max1 = game.X_raw[:,0].min()-.5,game.X_raw[:,0].max()+.5
        min2, max2 = game.X_raw[:,1].min()-.5,game.X_raw[:,1].max()+.5
        # define the x and y scale
        x_grid = np.arange(min1, max1, 0.02)
        y_grid = np.arange(min2, max2, 0.02)
        zz = PlotUncert2(x_grid,y_grid,game.clf,game.unc)
        setvis = game.unc_ol.get_visible()
        game.cbar.remove()
        game.unc_ol.remove()
        game.unc_ol = game.ax.contourf(x_grid, y_grid, zz, alpha=1, zorder=0)
        game.overlay_by_label["uncertainty"] = game.unc_ol
        game.unc_ol.set_visible(setvis)
        game.cbar = game.fig.colorbar(game.unc_ol)
        game.cbar.ax.set_ylabel('uncertainty')
        plt.draw()
    else:
        print("No data to train on yet.")

# Function highlights training data.
def callbutton2(event,game):
    if len(game.X_train)!=0:
        if not game.training_data_shown:
            drawn_with = [x for x in game.drawn_points if x[2] in game.training_indices]
            for pt in drawn_with:
                pt[1].set_facecolor("yellow")
            game.training_data_shown = True
        else:
            drawn_with = [x for x in game.drawn_points if x[2] in game.training_indices]
            for pt in drawn_with:
                pt[1].set_facecolor(props[game.y_raw[pt[2]]])
            game.training_data_shown = False
        plt.draw()
    else:
        print("No training data available yet.")

# Function for selecting the point with the highest uncertainty.
def callbutton3(event,game):
    if len(game.X_train)!=0:
        if not game.hint_shown:
            game.clf.fit(game.X_train, game.y_train)
            game.hint = game.X_pool[np.argmax(entr(game.clf.predict_proba(game.X_pool)).sum(axis=1))]
            #index = np.nonzero((list(game.points_dict.values()) == game.hint).all(axis=1))[0][0]
            game.hint_index = np.nonzero((list(game.points_dict.values()) == game.hint).all(axis=1))[0][0]
            game.drawn_points[game.hint_index][1].set_edgecolor("purple")
            game.drawn_points[game.hint_index][1].set_facecolor("yellow")
            game.arrow = game.ax.annotate('hint', game.hint, xytext=game.hint - np.array((.75,.75)),
                                          arrowprops=dict(facecolor='black', shrink=0.2),
                                          )
            game.hint_shown = True
        else:
            #index = np.nonzero((list(game.points_dict.values()) == game.hint).all(axis=1))[0][0]
            game.drawn_points[game.hint_index][1].set_edgecolor("face")
            #game.y_raw needs to be changed to predicted class
            game.drawn_points[game.hint_index][1].set_facecolor(props[game.clf.predict(game.drawn_points[game.hint_index][0].reshape(1,-1))[0]])
            game.arrow.remove()
            game.hint_shown = False
        plt.draw()
    else:
        print("No data to train on yet.")

def fincall(event,game):
    if len(game.X_al_train)!=0:
        if not game.al_training_data_shown:
            drawn_with = [x for x in game.drawn_points if x[2] in game.al_training_indices]
            for pt in drawn_with:
                pt[1].set_facecolor("pink")
                pt[1].set_edgecolor("purple")
            game.al_training_data_shown = True
        else:
            drawn_with = [x for x in game.drawn_points if x[2] in game.al_training_indices]
            for pt in drawn_with:
                pt[1].set_facecolor(props[game.y_raw[pt[2]]])
                if pt[2] in game.training_indices:
                    pt[1].set_edgecolor("black")
                else:
                    pt[1].set_edgecolor("face")
            game.al_training_data_shown = False
        plt.draw()
    else:
        print("No training data available yet.")

def model_choice(label,game):
    dict_class = {"Random Forest":"forest", "Gaussian Naive Bayes":"gauss", "Neural Network":"nn"}
    game.set_clf(dict_class[label])
    game.update_model()

def uncert_choice(label,game):
    dict_uncert = {"Entropy":"entropy", "Least Confidence":"lconf", "Smallest Margin":"marg"}
    game.unc = dict_uncert[label]
    #game.update_uncert() #to be implemented

def main(input="iris",classifier="forest",uncertainty="entropy"):
    global t
    t = datetime.datetime(1,1,1)

    #create list of dicts to use for coloring the points. When using a dataset with more classes this needs to be extended.
    global props 
    props = {0:"red", 1:"blue", 2:"green", 3:"orange", 4:"purple", 5:"brown", 6:"pink", 7:"gray"}

    game = game_state(input,classifier,uncertainty)

    # Make list of final accuracy
    global_history = 0

    # Create the checkboxes:
    rax = game.fig.add_axes([0.15, 0.05, 0.09, 0.075])
    check = mpl.widgets.CheckButtons(
        ax=rax,
        labels=game.overlay_by_label.keys(),
        actives=[l.get_visible() for l in game.overlay_by_label.values()],
    )

    # Create button for uncertainty recalculation.
    # There are two reasons for having a manual recalculation instead of doing it automatically after each point:
    # 1. The calculation is time-consuming, so if someone does not want to use the uncertainty measure,
    #    it unnecessarily slows down the process significantly.
    # 2. If you want to imitate "batch querying", it makes sense to use the same uncertainty measurements multiple times before recalculating.
    #    Note that this is not quite batch querying, as the label of a queried point still becomes visible right away.
    bax = game.fig.add_axes([0.45, 0.05, 0.09, 0.075])
    butt = mpl.widgets.Button(
        ax = bax,
        label = "Recalculate\n Uncertainty"
    )

    # Button for highlighting training data
    bax2 = game.fig.add_axes([0.35, 0.05, 0.09, 0.075])
    butt2 = mpl.widgets.Button(
        ax = bax2,
        label = "Toggle Show\n training data"
    )

    # Button for highlighting the point with the highest uncertainty, according to the model.
    bax3 = game.fig.add_axes([0.25, 0.05, 0.09, 0.075])
    butt3 = mpl.widgets.Button(
        ax = bax3,
        label = "Point w/ highest u"
    )

    # Button for highlighting AL points and finishing the game
    finbax = game.fig.add_axes([0.55, 0.05, 0.09, 0.075])
    finbutt = mpl.widgets.Button(
        ax = finbax,
        label = "Show AL points"
    )

    # radio_classifier_dict = {"Random Forest":"forest", "Gaussian Naive Bayes":"gauss", "Neural Network":"nn"}
    # rax = game.fig.add_axes([0.05, 0.05, 0.15, 0.075])
    # radio = mpl.widgets.RadioButtons(ax=rax, labels=list(radio_classifier_dict.keys()), active=list(radio_classifier_dict.values()).index(classifier))

    # radio_uncert_dict = {"Entropy":"entropy", "Least Confidence":"lconf", "Smallest Margin":"marg"}
    # rax = game.fig.add_axes([0.05, 0.15, 0.15, 0.075])
    # radio_unc = mpl.widgets.RadioButtons(ax=rax, labels=list(radio_uncert_dict.keys()), active=list(radio_uncert_dict.values()).index(uncertainty))
    
    # radio.on_clicked(lambda event: model_choice(event, game))
    # radio_unc.on_clicked(lambda event: uncert_choice(event, game))

    check.on_clicked(lambda event: callback(event, game))

    cid_1 = butt.on_clicked(lambda event: callbutton(event, game))
    cid_2 = butt2.on_clicked(lambda event: callbutton2(event, game))
    cid_3 = butt3.on_clicked(lambda event: callbutton3(event, game))
    finbutt.on_clicked(lambda event: fincall(event,game))

    plt.xlim(game.X_pool[:,0].min()-.5,game.X_pool[:,0].max()+.5)
    plt.ylim(game.X_pool[:,1].min()-.5,game.X_pool[:,1].max()+.5)

    plt.show()

    #One final training with all queried points for evaluation
    game.clf.fit(game.X_train, game.y_train)
    predictions = game.clf.predict(game.X_test)
    #Save accuracy of current seed in array for later use
    global_history = accuracy_score(predictions, game.y_test)

    game.clf.fit(game.X_al_train,game.y_al_train)
    predictions = game.clf.predict(game.X_test)
    #Save accuracy of current seed in array for later use
    al_accuracy = accuracy_score(predictions, game.y_test)

    # This can probably be done in a nicer way for the main program to sort out the details.
    return("Human performance:\n" +
           "Final accuracy: " + str(global_history)+"\n"+
           "Total number of points queried: "+str(game.N_QUERIES)+"\n"+
           "Classes discovered: "+str(np.unique(game.y_train).size)+"/"+str(np.unique(game.y_raw).size) + "\n\n" +
           "AL performance:\n"+
           "Final accuracy: " + str(al_accuracy)+"\n"+
           "Total number of points queried: "+str(game.N_QUERIES)+"\n"+
           "Classes discovered: "+str(np.unique(game.y_al_train).size)+"/"+str(np.unique(game.y_raw).size))

class game_state:
    def __init__(self, input="iris", clf="forest", unc="entropy",init_number = 3, training_indices = None, show_acc = True):
        if training_indices == None:
            self.training_indices = []
        else:
            self.training_indices = training_indices
        self.clf = RandomForestClassifier(max_depth = 4, min_samples_split=2, n_estimators = 200, random_state = 1)
        self.set_clf(clf)
        self.unc = unc
        self.N_QUERIES = 0
        # fixing the dataset:
        # load the data (this could come from the GUI program)
        if input == "iris":
            iris = load_iris()
            X_rew = iris['data']
            y_rew = iris['target']
        elif input == "glass":
            X_rew = np.genfromtxt('glass.data', delimiter = ',', skip_header=0)[:,1:-1]
            print(X_rew[15])
            y_names = np.genfromtxt('glass.data', dtype = str, delimiter = ',', skip_header=0)[:,-1]
            #Preprocess classes into numerical values:
            my_dict = {name : ix for ix, name in enumerate(np.unique(y_names))}
            y_rew = np.fromiter([my_dict[zi] for zi in y_names], dtype = int)
        elif input == "glass_small":
            X_rew = np.genfromtxt('glass.data', delimiter = ',', skip_header=0)[:,1:-1]
            y_names = np.genfromtxt('glass.data', dtype = str, delimiter = ',', skip_header=0)[:,-1]
            #Preprocess classes into numerical values:
            my_dict = {name : ix for ix, name in enumerate(np.unique(y_names))}
            y_raw = np.fromiter([my_dict[zi] for zi in y_names], dtype = int)
            used_classes = [1,2,3]
            data = np.concatenate((X_rew,y_raw.reshape(y_raw.size,1)),axis=-1)
            XY = np.array([d for d in data if d[-1] in used_classes])
            X_rew = XY[:,:-1]
            y_rew = XY[:,-1]
        elif input == "moons":
            # Use moons dataset with some small noise.
            moons = make_moons(n_samples=150, noise=0.1, random_state=42)
            X_rew = moons[0]
            y_names = moons[1]
            #Preprocess classes into numerical values:
            my_dict = {name : ix for ix, name in enumerate(np.unique(y_names))}
            y_rew = np.fromiter([my_dict[zi] for zi in y_names], dtype = int)
        else:
            print("Dataset unknown, defaulting to iris")
            iris = load_iris()
            X_rew = iris['data']
            y_rew = iris['target']
        pca = PCA(n_components=2)
        transformed_data = pca.fit_transform(X=X_rew)

        # Isolate the data we'll need for plotting.
        self.X_rew = np.column_stack((transformed_data[:, 0], transformed_data[:, 1]))
        # The train-test-split is used for determining the final accuracy.
        self.X_raw, self.X_test, self.y_raw, self.y_test = train_test_split(self.X_rew,y_rew,test_size=.5,stratify=y_rew,random_state=42)
        # pick the initial training data
        n_labeled_examples = self.X_raw.shape[0]
        self.training_indices=list(np.random.permutation(n_labeled_examples-1)[0:init_number])
        self.al_training_indices = self.training_indices.copy()
        self.X_train = self.X_raw[self.training_indices]
        self.y_train = self.y_raw[self.training_indices]
        self.X_al_train = self.X_raw[self.al_training_indices]
        self.y_al_train = self.y_raw[self.al_training_indices]
        self.X_pool = np.delete(self.X_raw, self.training_indices, axis=0)
        self.y_pool = np.delete(self.y_raw, self.training_indices, axis=0)
        self.X_al_pool = np.delete(self.X_raw, self.al_training_indices, axis=0)
        self.y_al_pool = np.delete(self.y_raw, self.al_training_indices, axis=0)

        #Draw initial unlabelled dataset:
        self.fig, self.ax = plt.subplots(figsize=(16,9))
        self.ax.set_title("Dataset")
        self.plot_accuracy = [] #convention should be: track pairs consisting of (#of current datapoint, accuracy)
        self.fig.subplots_adjust(bottom=0.2)
        self.cid = self.fig.canvas.mpl_connect("pick_event", lambda event: onpick(event, self))
        self.fig.canvas.mpl_connect('close_event', on_close)
        self.drawn_points = [] #create list to store artists to change their color at some point.
        self.points_dict = {} #create dictionary to access drawn points from their location
        for i,X in enumerate(self.X_raw):
            self.drawn_points.append([X,self.ax.scatter(X[0],X[1], picker=True, facecolor="Black"),i])
            self.points_dict[i] = X
            # create contour artists to refer to them from the beginning
        min1, max1 = self.X_pool[:,0].min()-.5,self.X_pool[:,0].max()+.5
        min2, max2 = self.X_pool[:,1].min()-.5,self.X_pool[:,1].max()+.5
        # define the x and y scale
        x_grid = np.arange(min1, max1, 0.02)
        y_grid = np.arange(min2, max2, 0.02)
        zz = np.zeros((y_grid.size,x_grid.size))
        #create contours for decision boundaries and uncertainty
        self.cont = self.ax.pcolormesh(x_grid, y_grid, zz, alpha=0, zorder=0)
        self.unc_ol = self.ax.contourf(x_grid, y_grid, zz, alpha=0, zorder=0)
        # Set not visible if someone wants to work without being spoiled
        self.cont.set_visible(False)
        self.unc_ol.set_visible(False)
        # Create dictionary to connect checkbox with actual overlay
        self.overlay_by_label = {"dec_boundary":self.cont, "uncertainty":self.unc_ol}
        # Create colorbar for uncertainty quantification
        self.cbar = self.fig.colorbar(self.unc_ol)
        self.cbar.ax.set_ylabel('uncertainty')
        self.textb_0 = self.fig.text(0.7,0.05,"Number of queries: ")
        self.textb_1 = self.fig.text(0.8,0.05,"0 \n ")
        self.hint_shown = False
        self.training_data_shown = False
        self.al_training_data_shown = False
        self.update_model()

    def set_clf(self,clf):
        self.model_dict = {"forest" : RandomForestClassifier(max_depth = 4, min_samples_split=2, n_estimators = 200, random_state = 1),
                          "gauss" : GaussianNB(),
                          "nn" : MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)}
        try:
            self.clf = self.model_dict[clf]
        except:
            print("classifier unknown, defaulting to random forest")
            self.clf = self.model_dict["forest"]

    def update_model(self):
        self.clf.fit(self.X_train, self.y_train)
        cmap_list = ['mistyrose','lightblue','lightgreen','moccasin','violet','peru','hotpink','snow']
        cmap = mpl.colors.ListedColormap(cmap_list[0:np.unique(self.y_raw).size])
        #xx, yy, zz = PlotContour(self.X_train,self.X_pool,self.clf)
        min1, max1 = self.X_raw[:,0].min()-.5,self.X_raw[:,0].max()+.5
        min2, max2 = self.X_raw[:,1].min()-.5,self.X_raw[:,1].max()+.5
        # define the x and y scale
        x_grid = np.arange(min1, max1, 0.02)
        y_grid = np.arange(min2, max2, 0.02)
        zz = PlotContour2(x_grid,y_grid,self.clf)
        # It seems that the "cleanest" (possibly only) way of updating the contourf is to completely remove it.
        # This also removes it from the dictionary used in the check-Button as well as the visibility setting.
        # (Hence, the next lines of code are longer than just one line)
        setvis = self.cont.get_visible()
        self.cont.remove()
        self.cont = self.ax.pcolormesh(x_grid, y_grid, zz, cmap=cmap, alpha=1, zorder=0, vmin=0, vmax=np.unique(self.y_raw).size-1, shading = "nearest")
        self.overlay_by_label["dec_boundary"] = self.cont
        self.cont.set_visible(setvis)
        plt.draw()
        if self.hint_shown:
            self.drawn_points[self.hint_index][1].set_edgecolor("face")
            self.drawn_points[self.hint_index][1].set_facecolor(props[self.clf.predict(self.drawn_points[self.hint_index][0].reshape(1,-1))[0]])
            self.arrow.remove()
            self.hint_shown = False
        drawn_without = [x for x in self.drawn_points if x[2] not in self.training_indices]
        drawn_with = [x for x in self.drawn_points if x[2] in self.training_indices]
        al_drawn_without = [x for x in self.drawn_points if x[2] not in self.al_training_indices]
        al_drawn_with = [x for x in self.drawn_points if x[2] in self.al_training_indices]
        for pt in drawn_without:
            pt[1].set_facecolor(props[self.clf.predict(pt[0].reshape(1,-1))[0]])
            pt[1].set_edgecolor("face")
            pt[1].set_alpha(.5)
        for pt in drawn_with:
            pt[1].set_facecolor(props[self.y_raw[pt[2]]])
            pt[1].set_alpha(1)
            pt[1].set_edgecolor("Black")
            pt[1].set_picker(None)
        self.textb_1.set_text(str(self.N_QUERIES))
        self.training_data_shown = False
        self.al_training_data_shown = False
        
def on_close(event):
    pass

if __name__ == "__main__":
    main()
from dataclasses import replace
import numpy as np
import json
from sklearn.datasets import load_iris, make_moons, make_circles, load_wine, load_digits
from sklearn.preprocessing import StandardScaler
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
import streamlit as st

def PlotContour2(x_grid, y_grid, clf):
    # This function generates a grid of function values according to the classifier.
    zz = np.zeros((y_grid.size,x_grid.size))
    for i, x in enumerate(x_grid):
        grid_array = [(x,yval) for yval in y_grid]
        zz[:,i] = clf.predict(grid_array)
    return zz

def PlotUncert2(x_grid,y_grid,clf,unc):
    # This function generates a grid of uncertainty values.
    # Note that the uncertainty values need to be calculated manually beforehand.
    # The "if"-branches select which uncertainty function to use.
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

    # ✅ Highlight the last clicked point
    if hasattr(game, 'last_clicked_artist') and game.last_clicked_artist is not None:
        game.last_clicked_artist.set_edgecolor("black")
        game.last_clicked_artist.set_linewidth(1.0)

    artist.set_edgecolor("yellow")   
    artist.set_linewidth(2.5)        # Thicker edge for visibility
    game.last_clicked_artist = artist

    selected_point = artist.get_offsets()
    query_index = np.where(game.X_pool == selected_point[0])[0][0]
    raw_index = np.where(game.X_raw == selected_point[0])[0][0]
    #game.human_selected_points.append(raw_index.item())
    # if game.citizen_science_mode:
    #     if hasattr(artist, "index"):
    #         game.human_selected_points.append(artist.index)
    #         print(f"[Citizen Science] Clicked point index: {raw_index.item()}, coords: {selected_point}")
    game.training_indices.append(raw_index.item())
    X, y = game.X_pool[query_index].reshape(1, -1), game.y_pool[query_index].reshape(1, )
    #add the new labeled point to the training data
    game.X_train = np.concatenate((game.X_train,X),axis=0)
    game.y_train = np.append(game.y_train,y)
    # Remove the queried instance from the unlabeled pool.
    game.X_pool, game.y_pool = np.delete(game.X_pool, query_index, axis=0), np.delete(game.y_pool, query_index)
    #artist.set_facecolor(props[y[0]])
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
        x_grid = np.arange(min1, max1, 0.05)
        y_grid = np.arange(min2, max2, 0.05)
        zz = PlotUncert2(x_grid,y_grid,game.clf,game.unc)
        setvis = game.unc_ol.get_visible()
        game.cbar.remove()
        game.unc_ol.remove()
        game.unc_ol = game.ax.contourf(x_grid, y_grid, zz, alpha=1, zorder=0)
        game.overlay_by_label["uncert"] = game.unc_ol
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
                 pt[1].set_edgecolor("black")   
                 pt[1].set_alpha(1)  
            game.training_data_shown = True
        else:
            drawn_with = [x for x in game.drawn_points if x[2] in game.training_indices]
            for pt in drawn_with:
                pt[1].set_facecolor(props[game.y_raw[pt[2]]])
            pt[1].set_alpha(1)  
            game.training_data_shown = False
        plt.draw()
    else:
        print("No training data available yet.")

# Function for highlighting the point with the highest uncertainty.
def callbutton3(event,game):
    if len(game.X_train)!=0:
        # if game.citizen_science_mode and game.human_selected_points:
        #     idx = game.citizen_science_query()
        #     print(">>> citizen_science_query CALLED <<<")
        #     if idx is not None:
        #         game.hint = game.X_pool[idx]
        #         return

        if not game.hint_shown:
            game.clf.fit(game.X_train, game.y_train)
            game.hint = game.X_pool[np.argmax(entr(game.clf.predict_proba(game.X_pool)).sum(axis=1))]
            #index = np.nonzero((list(game.points_dict.values()) == game.hint).all(axis=1))[0][0]
            game.hint_index = np.nonzero((list(game.points_dict.values()) == game.hint).all(axis=1))[0][0]
            # Find index of closest match to suggested coordinates
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

# Function executes whenever the radio button "radio" is changed.
def model_choice(label,game):
    dict_class = {"Random Forest":"forest", "Gaussian Naive Bayes":"gauss", "Neural Network":"nn"}
    game.set_clf(dict_class[label])
    game.update_model()

# Function executes whenever the radio button "radio_unc" is changed.
def uncert_choice(label,game):
    dict_uncert = {"Entropy":"entropy", "Least Confidence":"lconf", "Smallest Margin":"marg"}
    game.unc = dict_uncert[label]
    #game.update_uncert() #to be implemented

def debug_printer(game):
    st.write("It's working! It's working!")

def main(classifier="forest",uncertainty="entropy", extra_params={}):
    st.write("main initialized!")
    global t
    t = datetime.datetime(1,1,1)
    
    # create list of dicts to use for coloring the points. When using a dataset with more classes this needs to be extended.
    # The Iris dataset only uses the first three, but if we use datasets with a higher number of classes, we have it available here.
    global props 
    props = {0:"red", 1:"blue", 2:"green", 3:"orange", 4:"purple", 5:"brown", 6:"pink", 7:"gray"}
    
    # Create game state
    game = game_state(input,classifier,uncertainty, extra_params=extra_params)

    # Make list of final accuracy
    global_history = 0

    # Create the checkboxes:
    rax = game.fig.add_axes([0.25, 0.15, 0.09, 0.075])
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
    #bax = game.fig.add_axes([0.45, 0.05, 0.09, 0.075])

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

    # Radio button for choosing the classifier
    radio_classifier_dict = {"Random Forest":"forest", "Gaussian Naive Bayes":"gauss", "Neural Network":"nn"}
    rax = game.fig.add_axes([0.05, 0.05, 0.15, 0.075])
    radio = mpl.widgets.RadioButtons(ax=rax, labels=list(radio_classifier_dict.keys()), active=list(radio_classifier_dict.values()).index(classifier))

    # Radio button for choosing the uncertainty function
    radio_uncert_dict = {"Entropy":"entropy", "Least Confidence":"lconf", "Smallest Margin":"marg"}
    rax = game.fig.add_axes([0.05, 0.15, 0.15, 0.075])
    radio_unc = mpl.widgets.RadioButtons(ax=rax, labels=list(radio_uncert_dict.keys()), active=list(radio_uncert_dict.values()).index(uncertainty))
    
    # Connect all buttons to their respective functions
    radio.on_clicked(lambda event: model_choice(event, game))
    radio_unc.on_clicked(lambda event: uncert_choice(event, game))

    check.on_clicked(lambda event: callback(event, game))

    butt2.on_clicked(lambda event: callbutton2(event, game))
    butt3.on_clicked(lambda event: callbutton3(event, game))

    plt.xlim(game.X_raw[:,0].min()-0.5,game.X_raw[:,0].max()+0.5)
    plt.ylim(game.X_raw[:,1].min()-0.5,game.X_raw[:,1].max()+0.5)
    st.pyplot(game.fig)
    
    #st.button(label = "Recalc. Uncertainty", on_click = (lambda game: debug_printer(game)))
    st.button(label = "Recalc. Uncertainty", on_click = debug_printer(game=game))

    #One final training with all queried points for evaluation
    #game.clf.fit(game.X_train, game.y_train)
    #predictions = game.clf.predict(game.X_test)
    #Save accuracy of current seed in array for later use
    #global_history = accuracy_score(predictions, game.y_test)

    # This can probably be done in a nicer way for the main program to sort out the details.
    #return("Final accuracy: " + str(global_history)+"\n"+
    #       "Total number of points: "+str(game.N_QUERIES)+"\n"+
    #       "Classes discovered: "+str(np.unique(game.y_train).size)+"/"+str(np.unique(game.y_raw).size))

# The following defines the "game state", which is the core of the program and contains most of the information used throughout the program,
# such as chosen data points and the current choice of model and uncertainty.
class game_state:
    def __init__(self, clf="forest", unc="entropy", N_QUERIES=0, training_indices = None, show_acc = True, extra_params=None):
        global props
        # This training indices part is done differently in the competition mode.
        # I will probably delete of change this at some point if I have the time.
        if training_indices == None:
            self.training_indices = []
        else:
            self.training_indices = training_indices
        # The following line sets the initial classifier to random forest, the next one sets it to whatever is chosen beforehand. I think the next line can be deleted.
        self.clf = RandomForestClassifier(max_depth = 4, min_samples_split=2, n_estimators = 200, random_state = 1)
        self.set_clf(clf)
        # Set initial uncertainty function
        self.unc = unc
        self.N_QUERIES = N_QUERIES
        # Set initial dataset:
        iris = load_iris()
        X_rew = iris['data']
        y_rew = iris['target']

        pca = PCA(n_components=2, svd_solver='randomized')
        transformed_data = pca.fit_transform(X=X_rew)

        # Isolate the data we'll need for plotting.
        self.X_rew = np.column_stack((transformed_data[:, 0], transformed_data[:, 1]))
        # The train-test-split is used for determining the final accuracy.
        self.X_raw, self.X_test, self.y_raw, self.y_test = train_test_split(self.X_rew, y_rew, test_size=.5, stratify=y_rew, random_state=42)
        # pick the initial training data. Usually, training_indices=None and hence, this creates an empty list for the training data.
        self.X_train = self.X_raw[self.training_indices]
        self.y_train = self.y_raw[self.training_indices]
        self.X_pool = np.delete(self.X_raw, self.training_indices, axis=0)
        self.y_pool = np.delete(self.y_raw, self.training_indices, axis=0)

        # Draw initial unlabelled dataset:
        # ax describes the "game" itself, while ax2 is drawn to display the progress of the accuracy when choosing points.
        self.fig, (self.ax, self.ax2) = plt.subplots(1,2,figsize=(16,8),width_ratios=(2,1),)
        self.ax.set_title("Dataset")
        self.ax2.set_title("Accuracy")
        self.ax2.set_xlim(1,10)
        self.ax2.set_ylim(0,1)#I hope this fixes the limits, otherwise I can also set them each time.
        self.plot_accuracy = [] #convention should be: track pairs consisting of (#of current datapoint, accuracy)
        self.fig.subplots_adjust(bottom=0.3)
        self.cid = self.fig.canvas.mpl_connect("pick_event", lambda event: onpick(event, self))
        self.fig.canvas.mpl_connect('close_event', on_close)
        self.drawn_points = [] # create list to store artists to change their color at some point.
        self.points_dict = {} # create dictionary to access drawn points from their location. Numbering by X_raw
        # The following fills the list drawn_points. This associates each data point with a fixed index.
        # The association with a fixed index is needed, since the points will move from the set X_pool to the set X_train (if clicked), hence the indices of X_pool and X_train keep changing.
        for i,X in enumerate(self.X_raw):
            self.drawn_points.append([X,self.ax.scatter(X[0],X[1], picker=True, facecolor="Black"),i])
            self.points_dict[i] = X
        min1, max1 = self.X_pool[:,0].min()-.5,self.X_pool[:,0].max()+.5
        min2, max2 = self.X_pool[:,1].min()-.5,self.X_pool[:,1].max()+.5
        # define the x and y scale
        x_grid = np.arange(min1, max1, 0.05)
        y_grid = np.arange(min2, max2, 0.05)
        zz = np.zeros((y_grid.size,x_grid.size))
        # Create contours for decision boundaries and uncertainty
        self.cont = self.ax.pcolormesh(x_grid, y_grid, zz, alpha=0, zorder=0)
        self.unc_ol = self.ax.contourf(x_grid, y_grid, zz, alpha=0, zorder=0)
        # Set not visible if someone wants to work without being spoiled
        self.cont.set_visible(False)
        self.unc_ol.set_visible(False)
        # Create dictionary to connect checkbox with actual overlay
        self.overlay_by_label = {"dec_bound":self.cont, "uncert":self.unc_ol}
        # Create colorbar for uncertainty quantification
        self.cbar = self.fig.colorbar(self.unc_ol)
        self.cbar.ax.set_ylabel('uncertainty')
        self.textb_0 = self.fig.text(0.7,0.05,"Number of data points:\nAccuracy:")
        self.textb_1 = self.fig.text(0.82,0.05,"0 \n ")
        self.hint_shown = False
        self.training_data_shown = False

    # Function for choosing the classifier
    def set_clf(self,clf, extra_params=None):
        if extra_params is None:
            extra_params = {}

        if clf == "forest":
            self.clf = RandomForestClassifier(**extra_params, random_state=1)
        elif clf == "gauss":
            smoothing = extra_params.get("var_smoothing", 1e-4)
            self.clf = GaussianNB(var_smoothing=smoothing)
        elif clf == "nn":
            hidden_layer_sizes = (extra_params.get("neurons_per_layer", 16),) * extra_params.get("num_layers", 2)
            self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                 hidden_layer_sizes=hidden_layer_sizes,
                                 random_state=1)
        else:
         print("Classifier unknown, defaulting to random forest.")
         self.clf = RandomForestClassifier(**extra_params, random_state=1)

        print("[DEBUG] Model Parameters:", self.clf.get_params())

    # Function to update the model. This retrains the classifier and redraws all points.
    # This is mostly executed once a new point is clicked and added to the training data, but also when the classifier is changed during the program.
    def update_model(self):
        if isinstance(self.clf, GaussianNB):
            if len(np.unique(self.y_train)) > 1:
                self.clf.fit(self.X_train, self.y_train)
            else:
                print("Not enough class diversity to train the GaussianNB model. Using fallback predictions.")
                # Fallback predict when only one class is in training data
                single_class = int(np.array(self.y_train)[0])
                n_classes = len(np.unique(self.y_raw))
                self.clf.predict = lambda X: np.full((len(X),), single_class, dtype=int)
                self.clf.predict_proba = lambda X: np.tile(np.eye(n_classes)[single_class], (len(X), 1))
        else:
            self.clf.fit(self.X_train, self.y_train)

        cmap_list = ['mistyrose','lightblue','lightgreen','moccasin','violet','peru','hotpink','snow']
        cmap = mpl.colors.ListedColormap(cmap_list[0:np.unique(self.y_raw).size])
        # Remove old hint arrow if it exists
        # if hasattr(self, 'arrow'):
        #     self.arrow.remove()
        #cmap = mpl.colors.ListedColormap([cmap_list[i] for i in np.unique(self.y_raw)])
        #xx, yy, zz = PlotContour(self.X_train,self.X_pool,self.clf)
        min1, max1 = self.X_raw[:,0].min()-.5,self.X_raw[:,0].max()+.5
        min2, max2 = self.X_raw[:,1].min()-.5,self.X_raw[:,1].max()+.5
        # define the x and y scale
        x_grid = np.arange(min1, max1, 0.05)
        y_grid = np.arange(min2, max2, 0.05)
        zz = PlotContour2(x_grid,y_grid,self.clf)
        # It seems that the "cleanest" (possibly only) way of updating the contourf is to completely remove it.
        # This also removes it from the dictionary used in the check-Button as well as the visibility setting.
        # (Hence, the next lines of code are needed)
        setvis = self.cont.get_visible()
        self.cont.remove()
        self.cont = self.ax.pcolormesh(x_grid, y_grid, zz, cmap=cmap, alpha=1, zorder=0, vmin=0, vmax=np.unique(self.y_raw).size-1, shading = "nearest")
        self.overlay_by_label["dec_bound"] = self.cont
        self.cont.set_visible(setvis)
        plt.draw()
        # Delete the hint and arrow if it was visible before:
        if self.hint_shown:
            self.drawn_points[self.hint_index][1].set_edgecolor("face")
            self.drawn_points[self.hint_index][1].set_facecolor(props[self.clf.predict(self.drawn_points[self.hint_index][0].reshape(1,-1))[0]])
            self.arrow.remove()
            self.hint_shown = False
        # To redraw all points, we distinguish between training data (ground truth known) and pool data (ground truth unknown).
        # For training data, the points are colored according to their ground truth and opaque,
        # while for the pool data, the points are drawn according to their prediction and transparent for easier distinction.
        drawn_without = [x for x in self.drawn_points if x[2] not in self.training_indices]
        drawn_with = [x for x in self.drawn_points if x[2] in self.training_indices]
        for pt in drawn_without:
            pt[1].set_facecolor(props[self.clf.predict(pt[0].reshape(1,-1))[0]])
            pt[1].set_alpha(.5)
        for pt in drawn_with:
            pt[1].set_facecolor(props[self.y_raw[pt[2]]])
            pt[1].set_alpha(1)
            pt[1].set_edgecolor("black")
            pt[1].set_picker(None)
        # Track accuracy and plot in the accuracy graph:
        predictions = self.clf.predict(self.X_test)
        # Save accuracy of current seed in array for plotting
        self.plot_accuracy.append((self.N_QUERIES,accuracy_score(predictions, self.y_test)))
        # mpl doesn't recognize the x-values if naively written, so the following is needed.
        x = np.array(self.plot_accuracy)[:,0]
        y = np.array(self.plot_accuracy)[:,1]
        if self.N_QUERIES == 1:
            self.line, = self.ax2.plot(x,y)
        else:
            self.line.set_data(x,y)
            self.ax2.set_xlim(right=np.max((10,np.ceil(self.N_QUERIES/5)*5)))
        self.textb_1.set_text(str(self.N_QUERIES) + "\n" + str(np.round(self.plot_accuracy[-1][1],decimals=2)))

def on_close(event):
    pass

st.button("Start game", on_click=main)

if __name__ == "__main__":
    def main():  # your arguments here
        game = game_state()

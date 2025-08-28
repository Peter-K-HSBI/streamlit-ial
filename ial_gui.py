import tkinter as tk
from tkinter import ttk, messagebox
import oop_al_picker as ial
import oop_al_competition as comp

class IAL_GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("500x600")
        self.root.title("InterActive Learner")

        self.label = tk.Label(self.root, text="InterActive Learner", font=("Arial", 14))
        self.label.pack()

        self.button_sandbox = tk.Button(self.root, text="Start Sandbox", command=self.start_ial, height=3, width=20)
        self.button_sandbox.pack()

        self.button_comp = tk.Button(self.root, text="Start Competition", command=self.start_comp, height=3, width=20)
        self.button_comp.pack(pady=10)

        self.dropdown_frame = tk.Frame(self.root, pady=10, width=400)

        # --- Classifier Selection ---
        tk.Label(self.dropdown_frame, text="Model:").grid(row=0, column=0, padx=10, pady=5, sticky="E")
        self.selected_option_classifier = tk.StringVar(value="Random Forest")
        options_classifier = ["Random Forest", "Gaussian Naive Bayes", "Neural Network"]
        self.dropdown_classifier = ttk.Combobox(self.dropdown_frame, textvariable=self.selected_option_classifier,
                                                values=options_classifier, width=20, state="readonly")
        self.dropdown_classifier.grid(row=0, column=1, pady=5)
        self.dropdown_classifier.bind("<<ComboboxSelected>>", self.toggle_model_params)

        # --- Dataset Selection ---
        tk.Label(self.dropdown_frame, text="Dataset:").grid(row=1, column=0, padx=10, pady=5, sticky="E")
        self.selected_option_data = tk.StringVar(value="Iris")
        options_data = ["Iris", "Two Moons","Circles", "Wine","Digits"]
        self.dropdown_data = ttk.Combobox(self.dropdown_frame, textvariable=self.selected_option_data,
                                          values=options_data, width=20, state="readonly")
        self.dropdown_data.grid(row=1, column=1, pady=5)

        # --- Uncertainty Function Selection ---
        tk.Label(self.dropdown_frame, text="Uncertainty function:").grid(row=2, column=0, padx=10, pady=5, sticky="E")
        self.selected_option_uncert = tk.StringVar(value="Entropy")
        options_uncert = ["Entropy", "Least Confidence", "Smallest Margin"]
        self.dropdown_uncert = ttk.Combobox(self.dropdown_frame, textvariable=self.selected_option_uncert,
                                            values=options_uncert, width=20, state="readonly")
        self.dropdown_uncert.grid(row=2, column=1, pady=5)

        self.dropdown_frame.pack(pady=10)

        # --- RANDOM FOREST PARAM FRAME ---
        self.param_rf_frame = tk.LabelFrame(self.root, text="Random Forest Parameters", pady=10, padx=10)

        tk.Label(self.param_rf_frame, text="Number of Trees:").grid(row=0, column=0, padx=10, pady=5, sticky="E")
        self.selected_n_estimators = tk.StringVar(value="100")
        self.entry_n_estimators = ttk.Entry(self.param_rf_frame, textvariable=self.selected_n_estimators, width=22)
        self.entry_n_estimators.grid(row=0, column=1, pady=5)

        tk.Label(self.param_rf_frame, text="Max Depth:").grid(row=1, column=0, padx=10, pady=5, sticky="E")
        self.selected_max_depth = tk.StringVar(value="50")
        self.entry_max_depth = ttk.Entry(self.param_rf_frame, textvariable=self.selected_max_depth, width=22)
        self.entry_max_depth.grid(row=1, column=1, pady=5)

        tk.Label(self.param_rf_frame, text="Split Quality:").grid(row=2, column=0, padx=10, pady=5, sticky="E")
        self.selected_criterion = tk.StringVar(value="gini")
        self.dropdown_criterion = ttk.Combobox(self.param_rf_frame, textvariable=self.selected_criterion,
                                               values=["gini", "entropy"], width=20, state="readonly")
        self.dropdown_criterion.grid(row=2, column=1, pady=5)

        # --- GAUSSIAN NAIVE BAYES PARAM FRAME (NEW) ---
        self.param_gnb_frame = tk.LabelFrame(self.root, text="Gaussian Naive Bayes Parameters", pady=10, padx=10)

        tk.Label(self.param_gnb_frame, text="Var Smoothing:").grid(row=0, column=0, padx=10, pady=5, sticky="E")
        self.selected_var_smoothing = tk.StringVar(value="1e-3")
        self.entry_var_smoothing = ttk.Entry(self.param_gnb_frame, textvariable=self.selected_var_smoothing, width=22)
        self.entry_var_smoothing.grid(row=0, column=1, pady=5)

        # --- NEURAL NETWORK PARAM FRAME ---
        self.param_nn_frame = tk.LabelFrame(self.root, text="Neural Network Parameters", pady=10, padx=10)

        tk.Label(self.param_nn_frame, text="Number of Layers:").grid(row=0, column=0, padx=10, pady=5, sticky="E")
        self.selected_num_layers = tk.StringVar(value="2")
        self.entry_num_layers = ttk.Entry(self.param_nn_frame, textvariable=self.selected_num_layers, width=22)
        self.entry_num_layers.grid(row=0, column=1, pady=5)

        tk.Label(self.param_nn_frame, text="Neurons per Layer:").grid(row=1, column=0, padx=10, pady=5, sticky="E")
        self.selected_neurons = tk.StringVar(value="16")
        self.entry_neurons = ttk.Entry(self.param_nn_frame, textvariable=self.selected_neurons, width=22)
        self.entry_neurons.grid(row=1, column=1, pady=5)

        # --- OUTPUT ---
        self.label2 = tk.Label(self.root, font=("Arial", 12), wraplength=480)
        self.label2.pack(pady=10)

        self.toggle_model_params()
        self.root.mainloop()

    def toggle_model_params(self, event=None):
        model = self.selected_option_classifier.get()
        self.param_rf_frame.pack_forget()
        self.param_nn_frame.pack_forget()
        self.param_gnb_frame.pack_forget()  # hide all

        if model == "Random Forest":
            self.param_rf_frame.pack(pady=5)
        elif model == "Neural Network":
            self.param_nn_frame.pack(pady=5)
        elif model == "Gaussian Naive Bayes":
            self.param_gnb_frame.pack(pady=5)

    def start_ial(self):
        dict_class = {"Random Forest": "forest", "Gaussian Naive Bayes": "gauss", "Neural Network": "nn"}
        dict_data = {"Iris": "iris", "Glass Identification": "glass", "Glass Id. Small": "glass_small", "Two Moons": "moons", "Circles": "circles", "Wine": "wine","Digits": "digits"}
        dict_uncert = {"Entropy": "entropy", "Least Confidence": "lconf", "Smallest Margin": "marg"}

        model = self.selected_option_classifier.get()
        extra_params = {}

        if model == "Random Forest":
            try:
                n_estimators = int(self.selected_n_estimators.get())
                if n_estimators <= 0:
                    raise ValueError
                extra_params["n_estimators"] = n_estimators
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a positive integer for Number of Trees.")
                self.selected_n_estimators.set("100")
                return

            try:
                max_depth_val = self.selected_max_depth.get()
                if max_depth_val.lower() == "none":
                    extra_params["max_depth"] = None
                else:
                    max_depth = int(max_depth_val)
                    if max_depth <= 0:
                        raise ValueError
                    extra_params["max_depth"] = max_depth
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a positive integer or 'None' for Max Depth.")
                self.selected_max_depth.set("50")
                return

            extra_params["criterion"] = self.selected_criterion.get()

        elif model == "Gaussian Naive Bayes":
            try:
                var_smoothing = float(self.selected_var_smoothing.get())
                if var_smoothing <= 0:
                    raise ValueError
                extra_params["var_smoothing"] = var_smoothing
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a positive float for Var Smoothing.")
                self.selected_var_smoothing.set("1e-3")
                return

        elif model == "Neural Network":
            try:
                num_layers = int(self.selected_num_layers.get())
                neurons = int(self.selected_neurons.get())
                if num_layers <= 0 or neurons <= 0:
                    raise ValueError
                extra_params["num_layers"] = num_layers
                extra_params["neurons_per_layer"] = neurons
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter positive integers for Neural Network parameters.")
                self.selected_num_layers.set("2")
                self.selected_neurons.set("16")
                return

        result = ial.main(
            dict_data[self.selected_option_data.get()],
            dict_class[self.selected_option_classifier.get()],
            dict_uncert[self.selected_option_uncert.get()],
        )
        self.label2['text'] = str(result)

    def start_comp(self):
        dict_class = {"Random Forest": "forest", "Gaussian Naive Bayes": "gauss", "Neural Network": "nn"}
        dict_data = {"Iris": "iris", "Glass Identification": "glass", "Glass Id. Small": "glass_small", "Two Moons": "moons","Circles": "circles", "Wine": "wine", "Digits": "digits"}
        dict_uncert = {"Entropy": "entropy", "Least Confidence": "lconf", "Smallest Margin": "marg"}

        model = self.selected_option_classifier.get()
        extra_params = {}

        try:
            if model == "Random Forest":
                n_estimators = int(self.selected_n_estimators.get())
                if n_estimators <= 0:
                    raise ValueError
                extra_params["n_estimators"] = n_estimators

                max_depth_val = self.selected_max_depth.get()
                if max_depth_val.lower() == "none":
                    extra_params["max_depth"] = None
                else:
                    max_depth = int(max_depth_val)
                    if max_depth <= 0:
                        raise ValueError
                    extra_params["max_depth"] = max_depth

                extra_params["criterion"] = self.selected_criterion.get()

            elif model == "Gaussian Naive Bayes":
                var_smoothing = float(self.selected_var_smoothing.get())
                if var_smoothing <= 0:
                    raise ValueError
                extra_params["var_smoothing"] = var_smoothing

            elif model == "Neural Network":
                num_layers = int(self.selected_num_layers.get())
                neurons = int(self.selected_neurons.get())
                if num_layers <= 0 or neurons <= 0:
                    raise ValueError
                extra_params["num_layers"] = num_layers
                extra_params["neurons_per_layer"] = neurons

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid positive numbers for model parameters.")
            return

        result = comp.main(
            dict_data[self.selected_option_data.get()],
            dict_class[self.selected_option_classifier.get()],
            dict_uncert[self.selected_option_uncert.get()],
        )
        self.label2['text'] = str(result)

IAL_GUI()

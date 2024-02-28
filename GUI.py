import tkinter as tk
from tkinter import filedialog, messagebox
# Assuming RBM and DataSet are defined as in your provided code
from RBM import RBM
from DataSet import DataSet


class RBMGUI:
    def __init__(self, master):
        self.master = master
        master.title("RBM GUI")
        master.geometry("700x500")
        master.configure(bg='white', padx=20, pady=20)

        # Initialize RBM with empty data initially
        self.rbm = RBM(DataSet())

        # Styling
        button_style = {'width': 20, 'height': 3,
                        'bg': "lightblue", 'fg': "black", 'font': ("Arial", 14)}

        # Data Management Frame
        data_frame = tk.Frame(master, bg='white')
        data_frame.pack(pady=20)

        # Load Data Button
        self.load_data_btn = tk.Button(
            data_frame, text="Load Data-set", command=self.load_data, **button_style, foreground="red")
        self.load_data_btn.grid(row=0, column=0, padx=10)

        # Load Model Button
        self.load_model_btn = tk.Button(
            data_frame, text="Load Full Model", command=self.load_model, **button_style)
        self.load_model_btn.grid(row=0, column=1, padx=10)

        # Model Training and Testing Frame
        train_test_frame = tk.Frame(master, bg='white')
        train_test_frame.pack(pady=20)

        # Train Button
        self.train_btn = tk.Button(
            train_test_frame, text="Train Model", command=self.train_model, **button_style)
        self.train_btn.grid(row=0, column=0, padx=10)

        # Test Button
        self.test_btn = tk.Button(
            train_test_frame, text="Test Model", command=self.test_model, **button_style)
        self.test_btn.grid(row=0, column=1, padx=10)

        # Model Saving and Loading Frame
        save_load_frame = tk.Frame(master, bg='white')
        save_load_frame.pack(pady=20)

        # Save Model Button
        self.save_model_btn = tk.Button(
            save_load_frame, text="Save Model", command=self.save_model, **button_style)
        self.save_model_btn.grid(row=0, column=0, padx=10)

        # Visualization Frame
        visualization_frame = tk.Frame(master, bg='white')
        visualization_frame.pack(pady=20)

        # Plot Network Button
        self.plot_network_btn = tk.Button(
            visualization_frame, text="Plot Network", command=self.plot_network, **button_style)
        self.plot_network_btn.grid(row=0, column=0, padx=10)

        # Plot Weights Button
        self.plot_weights_btn = tk.Button(
            visualization_frame, text="Plot Weights", command=self.plot_weights, **button_style)
        self.plot_weights_btn.grid(row=0, column=1, padx=10)

        # Plot Bias Button
        self.plot_bias_btn = tk.Button(
            visualization_frame, text="Plot Bias", command=self.plot_bias, **button_style)
        self.plot_bias_btn.grid(row=0, column=2, padx=10)

        # Load Weights Button
        self.load_weights_btn = tk.Button(
            visualization_frame, text="Load Weights", command=self.load_weights, **button_style)
        self.load_weights_btn.grid(row=1, column=0, padx=10)

        # Load Visible Bias Button
        self.load_visible_bias_btn = tk.Button(
            visualization_frame, text="Load Visible Bias", command=self.load_visible_bias, **button_style)
        self.load_visible_bias_btn.grid(row=1, column=1, padx=10)

        # Load Hidden Bias Button
        self.load_hidden_bias_btn = tk.Button(
            visualization_frame, text="Load Hidden Bias", command=self.load_hidden_bias, **button_style)
        self.load_hidden_bias_btn.grid(row=1, column=2, padx=10)


    def load_model(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            try:
                self.rbm.load_model(dir_path)
                messagebox.showinfo("Load Model", "Model loaded successfully!")
                # change color and text of the button
                self.load_model_btn.config(foreground="green", text="Model Loaded")
            except Exception as e:
                messagebox.showerror("Load Model", f"Failed to load model: {e}")

    def plot_bias(self):
        try:
            self.rbm.plotBias()
        except Exception as e:
            messagebox.showerror("Plot Bias", f"Failed to plot bias: {e}")

    def load_data(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                data = DataSet.createDataSet(file_path)
                file_path = file_path.split("/")[-1]
                self.rbm.setDataSet(data)
                messagebox.showinfo("Load Data", "Data loaded successfully!")
                # change color and text of the button
                self.load_data_btn.config(foreground="green", text=f'{file_path} Loaded')
            except Exception as e:
                messagebox.showerror("Load Data", f"Failed to load data: {e}")

    def train_model(self):
        try:
            self.rbm.train(epochs=32)
            messagebox.showinfo("Train Model", "Model trained successfully!")
        except Exception as e:
            messagebox.showerror("Train Model", f"Failed to train model: {e}")

    def test_model(self):
        try:
            stats = self.rbm.test()
            accuracy = 0
            total_instances = 0
            stats_message = ""
            for flower_type, data in stats.items():
                correct = data['correct']
                total = data['total']
                accuracy += correct
                total_instances += total
                stats_message += f"For {flower_type}, {correct} out of {total}.\n"
            accuracy = accuracy / total_instances
            stats_message += f"\nOverall accuracy: {accuracy * 100:.2f}%"
            messagebox.showinfo("Test Model", "Model tested successfully!")
            messagebox.showinfo("Test Results", stats_message)
        except Exception as e:
            messagebox.showerror("Test Model", f"An error occurred: {str(e)}")

    def save_model(self):
        directory = filedialog.askdirectory()
        if directory:
            try:
                self.rbm.save_model(directory)
                messagebox.showinfo("Save Model", "Model saved successfully!")
            except Exception as e:
                messagebox.showerror(
                    "Save Model", f"Failed to save model: {e}")

    def load_weights(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                self.rbm.load_weights(file_path)
                messagebox.showinfo("Load Weights", "Weights loaded successfully!")
            except Exception as e:
                messagebox.showerror("Load Weights", f"Failed to load weights: {e}")

    def load_visible_bias(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                self.rbm.load_visible_bias(file_path)
                messagebox.showinfo("Load Visible Bias", "Visible bias loaded successfully!")
            except Exception as e:
                messagebox.showerror("Load Visible Bias", f"Failed to load visible bias: {e}")

    def load_hidden_bias(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                self.rbm.load_hidden_bias(file_path)
                messagebox.showinfo("Load Hidden Bias", "Hidden bias loaded successfully!")
            except Exception as e:
                messagebox.showerror("Load Hidden Bias", f"Failed to load hidden bias: {e}")

    def plot_weights(self):
        try:
            self.rbm.plotWeights()
        except Exception as e:
            messagebox.showerror("Plot Weights", f"Failed to plot weights: {e}")



    def plot_network(self):
        try:
            self.rbm.plotNetwork()
        except Exception as e:
            messagebox.showerror(
                "Plot Network", f"Failed to plot network: {e}")


def main():
    root = tk.Tk()
    gui = RBMGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

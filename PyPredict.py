import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import MaxNLocator

class PopulationPredictor:
    def __init__(self):
        self.years = []
        self.population = []
        self.newton_coeffs = []
        self.lr_model = None
        
    def load_data_from_input(self, years, population):
        """Muat data dari input pengguna"""
        self.years = np.array(years, dtype=float)
        self.population = np.array(population, dtype=float)
        
    def load_data_from_csv(self, file_path):
        """Muat data dari file CSV"""
        try:
            df = pd.read_csv(file_path)
            # Cek apakah kolom tahun dan populasi ada di file
            if len(df.columns) >= 2:
                self.years = np.array(df.iloc[:, 0], dtype=float)
                self.population = np.array(df.iloc[:, 1], dtype=float)
                return True, "Data berhasil dimuat"
            else:
                return False, "Format CSV tidak valid. Harap gunakan format: tahun,populasi"
        except Exception as e:
            return False, f"Error saat membaca file CSV: {e}"
    
    def _divided_difference(self, x, y):
        """Hitung koefisien beda terbagi untuk Interpolasi Newton"""
        n = len(y)
        coeffs = np.zeros([n, n])
        coeffs[:, 0] = y
        
        for j in range(1, n):
            for i in range(n - j):
                coeffs[i, j] = (coeffs[i+1, j-1] - coeffs[i, j-1]) / (x[i+j] - x[i])
                
        return coeffs[0, :]
    
    def train_newton_interpolation(self):
        """Latih model Interpolasi Newton"""
        if len(self.years) < 2:
            return False, "Butuh setidaknya 2 titik data untuk interpolasi"
        
        self.newton_coeffs = self._divided_difference(self.years, self.population)
        return True, "Model Interpolasi Newton berhasil dilatih"
    
    def predict_newton(self, x):
        """Prediksi dengan Interpolasi Newton"""
        n = len(self.years)
        p = self.newton_coeffs[0]
        
        for i in range(1, n):
            term = self.newton_coeffs[i]
            for j in range(i):
                term *= (x - self.years[j])
            p += term
            
        return p
    
    def train_linear_regression(self):
        """Latih model Regresi Linier Sederhana"""
        if len(self.years) < 2:
            return False, "Butuh setidaknya 2 titik data untuk regresi"
        
        X = self.years.reshape(-1, 1)
        y = self.population
        
        self.lr_model = LinearRegression()
        self.lr_model.fit(X, y)
        return True, "Model Regresi Linier berhasil dilatih"
    
    def predict_linear_regression(self, x):
        """Prediksi dengan Regresi Linier Sederhana"""
        if isinstance(x, (int, float)):
            return self.lr_model.predict([[x]])[0]
        else:
            return self.lr_model.predict(np.array(x).reshape(-1, 1))
    
    def compare_methods(self, target_years):
        """Bandingkan hasil prediksi kedua metode"""
        if not isinstance(target_years, list):
            target_years = [target_years]
            
        results = []
        for year in target_years:
            newton_pred = self.predict_newton(year)
            lr_pred = self.predict_linear_regression(year)
            results.append({
                'Tahun': year,
                'Newton': newton_pred,
                'Regresi': lr_pred,
                'Selisih': abs(newton_pred - lr_pred)
            })
        
        return results
    
    def plot_predictions(self, fig, ax, future_years=None):
        """Visualisasikan data dan prediksi"""
        if future_years is None:
            # Default: prediksi 5 tahun ke depan dari tahun terakhir
            max_year = max(self.years)
            future_years = np.arange(max_year + 1, max_year + 6)
        
        # Buat data untuk plot
        all_years = np.concatenate([self.years, future_years])
        years_range = np.linspace(min(all_years), max(all_years), 100)
        
        # Prediksi untuk range tahun untuk kurva halus
        newton_predictions = np.array([self.predict_newton(yr) for yr in years_range])
        lr_predictions = self.predict_linear_regression(years_range)
        
        # Prediksi untuk tahun-tahun spesifik di masa depan
        future_newton = np.array([self.predict_newton(yr) for yr in future_years])
        future_lr = self.predict_linear_regression(future_years)
        
        # Plot
        ax.clear()
        
        # Data asli
        ax.scatter(self.years, self.population, color='black', label='Data Asli', s=50, zorder=5)
        
        # Kurva prediksi
        ax.plot(years_range, newton_predictions, 'r-', label='Interpolasi Newton', linewidth=2)
        ax.plot(years_range, lr_predictions, 'b-', label='Regresi Linier', linewidth=2)
        
        # Titik prediksi di masa depan
        ax.scatter(future_years, future_newton, color='red', marker='x', s=100, label='Prediksi Newton', zorder=4)
        ax.scatter(future_years, future_lr, color='blue', marker='+', s=100, label='Prediksi Regresi Linier', zorder=4)
        
        # Tambahkan label pada titik prediksi
        for i, yr in enumerate(future_years):
            ax.annotate(f"{int(future_newton[i]):,}", (yr, future_newton[i]), 
                        textcoords="offset points", xytext=(0,10), ha='center', color='red')
            ax.annotate(f"{int(future_lr[i]):,}", (yr, future_lr[i]), 
                        textcoords="offset points", xytext=(0,-15), ha='center', color='blue')
        
        # Label dan judul
        ax.set_title('Prediksi Pertumbuhan Populasi', fontsize=12)
        ax.set_xlabel('Tahun', fontsize=10)
        ax.set_ylabel('Populasi', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=9)
        
        # Format sumbu x agar menampilkan tahun sebagai bilangan bulat
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Format sumbu y untuk angka besar
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,}"))
        
        fig.tight_layout()
        return fig


class DataTable(tk.Frame):
    def __init__(self, parent, headers):
        super().__init__(parent)
        self.headers = headers
        self._create_widgets()
        
    def _create_widgets(self):
        # Tabel data
        self.tree = ttk.Treeview(self, columns=self.headers, show="headings")
        
        # Set header
        for header in self.headers:
            self.tree.heading(header, text=header)
            self.tree.column(header, width=100, anchor=tk.CENTER)
        
        # Scrollbar
        yscrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscrollbar.set)
        
        # Layout
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def clear(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
            
    def insert_data(self, data):
        self.clear()
        for idx, row in enumerate(data):
            values = [row[header] if isinstance(row[header], str) else 
                     f"{int(row[header]):,}" if row[header].is_integer() else 
                     f"{row[header]:.2f}" for header in self.headers]
            self.tree.insert("", tk.END, values=values)

            
class PopulationPredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Aplikasi Prediksi Pertumbuhan Populasi")
        self.geometry("1200x800")
        self.resizable(True, True)
        
        self.predictor = PopulationPredictor()
        self.data_loaded = False
        
        self._create_widgets()
        self._load_sample_data()  # Load data contoh secara default
        
    def _create_widgets(self):
        # Frame utama
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame kiri: input dan operasi
        left_frame = ttk.LabelFrame(main_frame, text="Input & Operasi")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        # 1. Input data section
        input_frame = ttk.LabelFrame(left_frame, text="Input Data")
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Tombol input
        btn_input_manual = ttk.Button(input_frame, text="Input Manual", command=self._show_manual_input)
        btn_input_manual.pack(fill=tk.X, padx=5, pady=2)
        
        btn_input_csv = ttk.Button(input_frame, text="Import CSV", command=self._import_csv)
        btn_input_csv.pack(fill=tk.X, padx=5, pady=2)
        
        btn_input_sample = ttk.Button(input_frame, text="Data Contoh", command=self._load_sample_data)
        btn_input_sample.pack(fill=tk.X, padx=5, pady=2)
        
        # 2. Tabel data input
        self.data_table_frame = ttk.LabelFrame(left_frame, text="Data Populasi")
        self.data_table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.data_table = DataTable(self.data_table_frame, ["Tahun", "Populasi"])
        self.data_table.pack(fill=tk.BOTH, expand=True)
        
        # 3. Prediksi
        predict_frame = ttk.LabelFrame(left_frame, text="Prediksi")
        predict_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Single year prediction
        single_year_frame = ttk.Frame(predict_frame)
        single_year_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(single_year_frame, text="Tahun Prediksi:").pack(side=tk.LEFT, padx=5)
        self.single_year_entry = ttk.Entry(single_year_frame, width=10)
        self.single_year_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(single_year_frame, text="Prediksi", command=self._predict_single_year).pack(side=tk.LEFT, padx=5)
        
        # Range prediction
        range_frame = ttk.Frame(predict_frame)
        range_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(range_frame, text="Dari:").pack(side=tk.LEFT, padx=5)
        self.start_year_entry = ttk.Entry(range_frame, width=8)
        self.start_year_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(range_frame, text="Sampai:").pack(side=tk.LEFT, padx=5)
        self.end_year_entry = ttk.Entry(range_frame, width=8)
        self.end_year_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(range_frame, text="Step:").pack(side=tk.LEFT, padx=5)
        self.step_entry = ttk.Entry(range_frame, width=5)
        self.step_entry.insert(0, "1")
        self.step_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(range_frame, text="Prediksi Range", command=self._predict_range).pack(side=tk.LEFT, padx=5)
        
        # 4. Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Siap. Silakan input data atau gunakan data contoh.")
        status_bar = ttk.Label(left_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Frame kanan: visualisasi dan hasil
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 1. Hasil prediksi
        results_frame = ttk.LabelFrame(right_frame, text="Hasil Prediksi")
        results_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.results_table = DataTable(results_frame, ["Tahun", "Newton", "Regresi", "Selisih"])
        self.results_table.pack(fill=tk.X, pady=5)
        
        # 2. Visualisasi
        viz_frame = ttk.LabelFrame(right_frame, text="Visualisasi")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, viz_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _show_manual_input(self):
        """Tampilkan dialog untuk input data manual"""
        input_dialog = tk.Toplevel(self)
        input_dialog.title("Input Data Manual")
        input_dialog.geometry("400x500")
        input_dialog.transient(self)
        input_dialog.grab_set()
        
        # Frame untuk input
        entry_frame = ttk.Frame(input_dialog)
        entry_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tabel untuk entri data
        columns = ("tahun", "populasi")
        tree = ttk.Treeview(entry_frame, columns=columns, show="headings")
        tree.heading("tahun", text="Tahun")
        tree.heading("populasi", text="Populasi")
        tree.column("tahun", width=100, anchor=tk.CENTER)
        tree.column("populasi", width=100, anchor=tk.CENTER)
        tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Frame untuk input single row
        row_frame = ttk.Frame(entry_frame)
        row_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(row_frame, text="Tahun:").pack(side=tk.LEFT, padx=5)
        year_entry = ttk.Entry(row_frame, width=10)
        year_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row_frame, text="Populasi:").pack(side=tk.LEFT, padx=5)
        pop_entry = ttk.Entry(row_frame, width=10)
        pop_entry.pack(side=tk.LEFT, padx=5)
        
        # Button frame
        btn_frame = ttk.Frame(entry_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        def add_row():
            try:
                year = float(year_entry.get())
                pop = float(pop_entry.get())
                tree.insert("", tk.END, values=(year, pop))
                year_entry.delete(0, tk.END)
                pop_entry.delete(0, tk.END)
                year_entry.focus()
            except ValueError:
                messagebox.showerror("Error", "Tahun dan populasi harus berupa angka")
        
        def remove_selected():
            selected = tree.selection()
            if selected:
                tree.delete(selected)
        
        def save_data():
            years = []
            population = []
            
            for item in tree.get_children():
                values = tree.item(item, "values")
                years.append(float(values[0]))
                population.append(float(values[1]))
            
            if len(years) < 2:
                messagebox.showerror("Error", "Minimal 2 data diperlukan")
                return
            
            self.predictor.load_data_from_input(years, population)
            self._update_data_table()
            self._train_models()
            input_dialog.destroy()
        
        ttk.Button(btn_frame, text="Tambah", command=add_row).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Hapus", command=remove_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Simpan", command=save_data).pack(side=tk.RIGHT, padx=5)
    
    def _import_csv(self):
        """Import data dari file CSV"""
        file_path = filedialog.askopenfilename(
            title="Pilih File CSV",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if file_path:
            success, message = self.predictor.load_data_from_csv(file_path)
            if success:
                self._update_data_table()
                self._train_models()
            else:
                messagebox.showerror("Error", message)
    
    def _load_sample_data(self):
        """Muat data contoh"""
        # Data contoh: Populasi Indonesia 2010-2020
        years = [2010, 2012, 2014, 2016, 2018, 2020]
        population = [238.5, 245.1, 252.2, 258.7, 264.2, 270.2]  # dalam juta
        
        self.predictor.load_data_from_input(years, population)
        self._update_data_table()
        self._train_models()
    
    def _update_data_table(self):
        """Update tabel data dengan data terbaru"""
        data = [{"Tahun": year, "Populasi": pop} 
                for year, pop in zip(self.predictor.years, self.predictor.population)]
        self.data_table.insert_data(data)
        self.data_loaded = True
    
    def _train_models(self):
        """Latih model prediksi"""
        if len(self.predictor.years) < 2:
            self.status_var.set("Error: Minimal 2 data diperlukan untuk prediksi")
            return False
        
        success_newton, message_newton = self.predictor.train_newton_interpolation()
        success_lr, message_lr = self.predictor.train_linear_regression()
        
        if success_newton and success_lr:
            self.status_var.set("Model berhasil dilatih. Siap untuk prediksi.")
            return True
        else:
            self.status_var.set(f"Error: {message_newton}, {message_lr}")
            return False
    
    def _predict_single_year(self):
        """Prediksi untuk satu tahun spesifik"""
        if not self.data_loaded or not self._check_models():
            return
        
        try:
            year = float(self.single_year_entry.get())
            results = self.predictor.compare_methods(year)
            self.results_table.insert_data(results)
            
            # Update plot
            future_years = [year]
            self.predictor.plot_predictions(self.fig, self.ax, future_years)
            self.fig.canvas.draw()
            
            self.status_var.set(f"Prediksi untuk tahun {int(year) if year.is_integer() else year} berhasil")
        except ValueError:
            self.status_var.set("Error: Masukkan tahun dengan format yang valid")
            messagebox.showerror("Error", "Masukkan tahun dengan format yang valid")
    
    def _predict_range(self):
        """Prediksi untuk rentang tahun"""
        if not self.data_loaded or not self._check_models():
            return
        
        try:
            start_year = float(self.start_year_entry.get())
            end_year = float(self.end_year_entry.get())
            step = float(self.step_entry.get())
            
            if start_year > end_year:
                messagebox.showerror("Error", "Tahun awal harus lebih kecil dari tahun akhir")
                return
            
            future_years = np.arange(start_year, end_year + step, step)
            results = self.predictor.compare_methods(future_years)
            self.results_table.insert_data(results)
            
            # Update plot
            self.predictor.plot_predictions(self.fig, self.ax, future_years)
            self.fig.canvas.draw()
            
            self.status_var.set(f"Prediksi untuk tahun {start_year} - {end_year} berhasil")
        except ValueError:
            self.status_var.set("Error: Masukkan tahun dan step dengan format yang valid")
            messagebox.showerror("Error", "Masukkan tahun dan step dengan format yang valid")
    
    def _check_models(self):
        """Cek apakah model sudah dilatih"""
        if not hasattr(self.predictor, 'newton_coeffs') or len(self.predictor.newton_coeffs) == 0 or self.predictor.lr_model is None:
            self.status_var.set("Error: Model belum dilatih. Pastikan data sudah dimuat.")
            messagebox.showerror("Error", "Model belum dilatih. Pastikan data sudah dimuat.")
            return False
        return True


if __name__ == "__main__":
    app = PopulationPredictorApp()
    app.mainloop()
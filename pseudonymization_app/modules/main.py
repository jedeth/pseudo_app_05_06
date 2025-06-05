# Fichier : mainv2.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application de Pseudonymisation avec SpaCy
==========================================

Cette application permet de :
1. Entraîner des modèles SpaCy personnalisés avec de nouvelles entités NER
2. Générer automatiquement des données d'entraînement
3. Pseudonymiser et dépseudonymiser des textes

Auteur: Assistant IA & Vous!
Date: 2024
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import json
import threading
from pathlib import Path
from datetime import datetime

# Ajout du dossier modules au chemin Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Import des modules personnalisés
try:
    from data_generator import TrainingDataGenerator
    from model_trainer import SpacyModelTrainer
    from pseudonymizer import TextPseudonymizer
    from utils import AppUtils
except ImportError as e:
    print(f"Erreur d'import des modules: {e}")
    print("Assurez-vous que tous les modules sont présents dans le dossier 'modules'")
    sys.exit(1)

# ==============================================================================
# REMPLACEZ VOTRE CLASSE EXISTANTE PAR CELLE-CI
# ==============================================================================
class PseudonymizationApp:
    """
    Application principale de pseudonymisation
    
    Cette classe gère l'interface graphique complète et coordonne
    tous les modules de l'application pour offrir une expérience
    utilisateur fluide et intuitive.
    """
    
    def __init__(self, root):
        """
        Initialise l'application principale
        """
        self.root = root
        self.root.title("Application de Pseudonymisation - SpaCy NER")
        self.root.geometry("1200x800")
        
        # Variables de l'application
        self.selected_base_model = tk.StringVar(value="fr_core_news_sm")
        self.custom_entities = []
        self.training_data_path = ""
        self.trained_model_path = ""
        self.entity_files = {}
        self.generated_training_data = None
        self.pseudonymizer = None
        self.correspondence_file_path = ""
        self.training_in_progress = False
        
        # Initialisation des modules
        self.data_generator = TrainingDataGenerator()
        self.model_trainer = None
        self.utils = AppUtils()
        
        self.setup_ui()
        self.create_directories()
        
    def create_directories(self):
        """ Crée les dossiers nécessaires à l'application. """
        directories = ['models', 'data', 'config']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            
    def setup_ui(self):
        """ Configure l'interface utilisateur avec tous les onglets. """
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.create_config_tab()
        self.create_data_generation_tab()
        self.create_training_tab()
        self.create_pseudonymization_tab()
        self.create_depseudonymization_tab()
        
        self.status_bar = tk.Label(self.root, text="Prêt", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_config_tab(self):
        """ Crée l'onglet de configuration. """
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="1. Configuration")
        
        title_label = tk.Label(config_frame, text="Configuration du Modèle de Base", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        model_frame = ttk.LabelFrame(config_frame, text="Modèle SpaCy de Base")
        model_frame.pack(fill=tk.X, padx=20, pady=10)
        
        models = ["fr_core_news_sm", "fr_core_news_md", "fr_core_news_lg", "en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
        for model in models:
            rb = tk.Radiobutton(model_frame, text=model, variable=self.selected_base_model, value=model, font=("Arial", 10))
            rb.pack(anchor=tk.W, padx=10, pady=2)
        
        entities_frame = ttk.LabelFrame(config_frame, text="Entités NER Personnalisées")
        entities_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        add_entity_frame = tk.Frame(entities_frame)
        add_entity_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(add_entity_frame, text="Nom de l'entité:").pack(side=tk.LEFT)
        self.entity_entry = tk.Entry(add_entity_frame, width=30)
        self.entity_entry.pack(side=tk.LEFT, padx=5)
        
        add_button = tk.Button(add_entity_frame, text="Ajouter Entité", command=self.add_custom_entity, bg="#4CAF50", fg="white")
        add_button.pack(side=tk.LEFT, padx=5)
        
        self.entities_listbox = tk.Listbox(entities_frame, height=8)
        self.entities_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        remove_button = tk.Button(entities_frame, text="Supprimer Entité Sélectionnée", command=self.remove_custom_entity, bg="#f44336", fg="white")
        remove_button.pack(pady=5)
        
        validate_button = tk.Button(config_frame, text="Valider Configuration", command=self.validate_configuration, bg="#2196F3", fg="white", font=("Arial", 12))
        validate_button.pack(pady=20)
        
    def create_data_generation_tab(self):
        """ Crée l'onglet de génération de données. """
        data_gen_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_gen_frame, text="2. Génération de Données")
        
        title_label = tk.Label(data_gen_frame, text="Génération Automatique des Données d'Entraînement", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        import_frame = ttk.LabelFrame(data_gen_frame, text="Import des Listes de Termes")
        import_frame.pack(fill=tk.X, padx=20, pady=10)
        
        instructions = tk.Label(import_frame, text="Importez un fichier texte (.txt) pour chaque entité, avec un terme par ligne.", wraplength=800, justify=tk.LEFT)
        instructions.pack(padx=10, pady=5)
        
        add_file_button = tk.Button(import_frame, text="Ajouter Fichier de Termes", command=self.add_terms_file, bg="#4CAF50", fg="white")
        add_file_button.pack(pady=5)
        
        self.imported_files_list = tk.Listbox(import_frame, height=5)
        self.imported_files_list.pack(fill=tk.X, padx=10, pady=5)
        
        params_frame = ttk.LabelFrame(data_gen_frame, text="Paramètres de Génération")
        params_frame.pack(fill=tk.X, padx=20, pady=10)
        
        sentences_frame = tk.Frame(params_frame)
        sentences_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(sentences_frame, text="Nombre de phrases par terme:").pack(side=tk.LEFT)
        self.sentences_per_term = tk.IntVar(value=5)
        sentences_spinbox = tk.Spinbox(sentences_frame, from_=1, to=50, textvariable=self.sentences_per_term, width=10)
        sentences_spinbox.pack(side=tk.LEFT, padx=5)
        
        generate_button = tk.Button(data_gen_frame, text="Générer Données d'Entraînement", command=self.generate_training_data, bg="#FF9800", fg="white", font=("Arial", 12))
        generate_button.pack(pady=20)
        
        preview_frame = ttk.LabelFrame(data_gen_frame, text="Aperçu des Données Générées")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.preview_text = scrolledtext.ScrolledText(preview_frame, height=10, wrap=tk.WORD)
        self.preview_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_training_tab(self):
        """ Crée l'onglet d'entraînement. (Version nettoyée) """
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="3. Entraînement")
        
        title_label = tk.Label(training_frame, text="Fine-tuning du Modèle SpaCy", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        data_frame = ttk.LabelFrame(training_frame, text="Données d'Entraînement")
        data_frame.pack(fill=tk.X, padx=20, pady=10)
        
        data_buttons_frame = tk.Frame(data_frame)
        data_buttons_frame.pack(pady=10)
        
        load_data_button = tk.Button(data_buttons_frame, text="Charger Données", command=self.load_training_data, bg="#4CAF50", fg="white")
        load_data_button.pack(side=tk.LEFT, padx=5)
        
        test_model_button = tk.Button(data_buttons_frame, text="Tester Modèle", command=self.test_trained_model, bg="#607D8B", fg="white")
        test_model_button.pack(side=tk.LEFT, padx=5)
        
        self.data_status_label = tk.Label(data_frame, text="Aucune donnée chargée", fg="red")
        self.data_status_label.pack()
        
        params_frame = ttk.LabelFrame(training_frame, text="Paramètres d'Entraînement")
        params_frame.pack(fill=tk.X, padx=20, pady=10)
        
        epochs_frame = tk.Frame(params_frame)
        epochs_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(epochs_frame, text="Nombre d'époques:").pack(side=tk.LEFT)
        self.epochs_var = tk.IntVar(value=30)
        self.epochs_spinbox = tk.Spinbox(epochs_frame, from_=5, to=200, textvariable=self.epochs_var, width=10)
        self.epochs_spinbox.pack(side=tk.LEFT, padx=5)
        
        batch_frame = tk.Frame(params_frame)
        batch_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(batch_frame, text="Taille de batch:").pack(side=tk.LEFT)
        self.batch_size_var = tk.IntVar(value=8)
        self.batch_spinbox = tk.Spinbox(batch_frame, from_=1, to=32, textvariable=self.batch_size_var, width=10)
        self.batch_spinbox.pack(side=tk.LEFT, padx=5)
        
        self.train_button = tk.Button(training_frame, text="Commencer l'Entraînement", command=self.start_training, bg="#2196F3", fg="white", font=("Arial", 12, "bold"))
        self.train_button.pack(pady=20)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(training_frame, variable=self.progress_var, maximum=100, length=600)
        self.progress_bar.pack(pady=10)
        
        log_frame = ttk.LabelFrame(training_frame, text="Journal d'Entraînement")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.training_log = scrolledtext.ScrolledText(log_frame, height=8, wrap=tk.WORD, state='disabled')
        self.training_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_pseudonymization_tab(self):
        """ Crée l'onglet de pseudonymisation. """
        # Ce code est correct, pas de changement nécessaire
        pseudo_frame = ttk.Frame(self.notebook)
        self.notebook.add(pseudo_frame, text="4. Pseudonymisation")
        
        title_label = tk.Label(pseudo_frame, text="Pseudonymisation de Texte", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        model_frame = ttk.LabelFrame(pseudo_frame, text="Sélection du Modèle")
        model_frame.pack(fill=tk.X, padx=20, pady=10)
        
        select_model_button = tk.Button(model_frame, text="Sélectionner Modèle Entraîné", command=self.select_trained_model, bg="#4CAF50", fg="white")
        select_model_button.pack(pady=10)
        
        self.model_status_label = tk.Label(model_frame, text="Aucun modèle sélectionné", fg="red")
        self.model_status_label.pack()
        
        input_frame = ttk.LabelFrame(pseudo_frame, text="Texte à Pseudonymiser")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        input_buttons_frame = tk.Frame(input_frame)
        input_buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(input_buttons_frame, text="Importer Fichier", command=lambda: self.import_text_file(self.input_text), bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(input_buttons_frame, text="Effacer", command=lambda: self.input_text.delete(1.0, tk.END), bg="#FF5722", fg="white").pack(side=tk.LEFT, padx=5)
        
        self.input_text = scrolledtext.ScrolledText(input_frame, height=8, wrap=tk.WORD)
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        pseudo_button = tk.Button(pseudo_frame, text="Pseudonymiser", command=self.pseudonymize_text, bg="#FF9800", fg="white", font=("Arial", 12))
        pseudo_button.pack(pady=10)
        
        output_frame = ttk.LabelFrame(pseudo_frame, text="Texte Pseudonymisé")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        output_buttons_frame = tk.Frame(output_frame)
        output_buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(output_buttons_frame, text="Exporter Fichier", command=lambda: self.export_text_file(self.output_text, "texte_pseudonymise.txt"), bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(output_buttons_frame, text="Copier vers Dépseudonymisation", command=self.copy_to_depseudo, bg="#9C27B0", fg="white").pack(side=tk.LEFT, padx=5)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=8, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_depseudonymization_tab(self):
        """ Crée l'onglet de dépseudonymisation. """
        # Ce code est correct, pas de changement nécessaire
        depseudo_frame = ttk.Frame(self.notebook)
        self.notebook.add(depseudo_frame, text="5. Dépseudonymisation")
        
        title_label = tk.Label(depseudo_frame, text="Dépseudonymisation de Texte", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        corresp_frame = ttk.LabelFrame(depseudo_frame, text="Fichier de Correspondance")
        corresp_frame.pack(fill=tk.X, padx=20, pady=10)
        
        load_corresp_button = tk.Button(corresp_frame, text="Charger Fichier de Correspondance", command=self.load_correspondence_file, bg="#4CAF50", fg="white")
        load_corresp_button.pack(pady=10)
        
        self.corresp_status_label = tk.Label(corresp_frame, text="Aucun fichier chargé", fg="red")
        self.corresp_status_label.pack()
        
        pseudo_input_frame = ttk.LabelFrame(depseudo_frame, text="Texte Pseudonymisé")
        pseudo_input_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        pseudo_buttons_frame = tk.Frame(pseudo_input_frame)
        pseudo_buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(pseudo_buttons_frame, text="Importer Fichier", command=lambda: self.import_text_file(self.pseudo_input_text), bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(pseudo_buttons_frame, text="Effacer", command=lambda: self.pseudo_input_text.delete(1.0, tk.END), bg="#FF5722", fg="white").pack(side=tk.LEFT, padx=5)
        
        self.pseudo_input_text = scrolledtext.ScrolledText(pseudo_input_frame, height=8, wrap=tk.WORD)
        self.pseudo_input_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        depseudo_button = tk.Button(depseudo_frame, text="Dépseudonymiser", command=self.depseudonymize_text, bg="#9C27B0", fg="white", font=("Arial", 12))
        depseudo_button.pack(pady=10)
        
        depseudo_output_frame = ttk.LabelFrame(depseudo_frame, text="Texte Original Restauré")
        depseudo_output_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        restore_buttons_frame = tk.Frame(depseudo_output_frame)
        restore_buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(restore_buttons_frame, text="Exporter Fichier", command=lambda: self.export_text_file(self.depseudo_output_text, "texte_original.txt"), bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        
        self.depseudo_output_text = scrolledtext.ScrolledText(depseudo_output_frame, height=8, wrap=tk.WORD)
        self.depseudo_output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # ==============================================================================
    # SECTION DES MÉTHODES DE LA CLASSE
    # ==============================================================================
    
    def add_custom_entity(self):
        """ Ajoute une entité personnalisée à la liste. """
        entity = self.entity_entry.get().strip().upper()
        if entity and entity not in self.custom_entities:
            self.custom_entities.append(entity)
            self.entities_listbox.insert(tk.END, entity)
            self.entity_entry.delete(0, tk.END)
            self.update_status(f"Entité '{entity}' ajoutée")
        elif entity in self.custom_entities:
            messagebox.showwarning("Attention", f"L'entité '{entity}' existe déjà")
    
    def remove_custom_entity(self):
        """ Supprime l'entité sélectionnée. """
        selection = self.entities_listbox.curselection()
        if selection:
            entity = self.entities_listbox.get(selection[0])
            self.entities_listbox.delete(selection[0])
            self.custom_entities.remove(entity)
            self.update_status(f"Entité '{entity}' supprimée")
    
    def validate_configuration(self):
        """ Valide la configuration du modèle. """
        if not self.custom_entities:
            messagebox.showwarning("Configuration incomplète", "Veuillez ajouter au moins une entité personnalisée.")
            return
        
        messagebox.showinfo("Configuration validée", f"Modèle de base: {self.selected_base_model.get()}\nEntités: {', '.join(self.custom_entities)}")
        self.update_status("Configuration validée. Prêt pour la génération de données.")
    
    def add_terms_file(self):
        """ Ouvre un dialogue pour ajouter un fichier de termes pour une entité. """
        if not self.custom_entities:
            messagebox.showwarning("Configuration requise", "Veuillez d'abord définir vos entités personnalisées.")
            return
        
        filepath = filedialog.askopenfilename(title="Sélectionner un fichier de termes", filetypes=[("Fichiers texte", "*.txt")])
        if not filepath:
            return
        
        entity_dialog = EntitySelectionDialog(self.root, self.custom_entities)
        selected_entity = entity_dialog.result
        
        if selected_entity:
            self.entity_files[selected_entity] = filepath
            display_text = f"{selected_entity}: {Path(filepath).name}"
            self.imported_files_list.insert(tk.END, display_text)
            self.update_status(f"Fichier ajouté pour l'entité {selected_entity}")

    def generate_training_data(self):
        """ Lance la génération automatique des données d'entraînement. """
        if not self.entity_files:
            messagebox.showwarning("Fichiers manquants", "Veuillez ajouter des fichiers de termes pour vos entités.")
            return
        
        try:
            progress_dialog = ProgressDialog(self.root, "Génération des données d'entraînement...")
            self.generated_training_data, stats = self.data_generator.generate_training_data(
                self.entity_files,
                sentences_per_term=self.sentences_per_term.get()
            )
            progress_dialog.destroy()
            
            if self.generated_training_data:
                preview_text = self.data_generator.preview_training_data(self.generated_training_data, max_examples=5)
                self.preview_text.delete(1.0, tk.END)
                self.preview_text.insert(1.0, preview_text)
                
                if messagebox.askyesno("Sauvegarde", f"{len(self.generated_training_data)} exemples générés.\nVoulez-vous les sauvegarder ?"):
                    self.save_generated_data()
                self.update_status(f"Génération terminée : {len(self.generated_training_data)} exemples créés.")
                self.data_status_label.config(text=f"✅ {len(self.generated_training_data)} exemples prêts", fg="green")
            else:
                messagebox.showwarning("Génération échouée", "Aucune donnée n'a pu être générée.")
        except Exception as e:
            messagebox.showerror("Erreur de génération", f"Une erreur est survenue : {e}")
    
    def save_generated_data(self):
        """ Sauvegarde les données générées dans un fichier JSON. """
        try:
            saved_path = self.data_generator.save_training_data(self.generated_training_data)
            self.training_data_path = saved_path
            messagebox.showinfo("Sauvegarde réussie", f"Données sauvegardées dans :\n{saved_path}")
        except Exception as e:
            messagebox.showerror("Erreur de sauvegarde", f"Erreur : {e}")

    def load_training_data(self):
        """ Charge des données d'entraînement depuis un fichier JSON. """
        filepath = filedialog.askopenfilename(title="Charger un fichier de données JSON", filetypes=[("Fichiers JSON", "*.json")])
        if not filepath:
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            if not isinstance(json_data, list):
                raise ValueError("Le fichier JSON doit contenir une liste d'exemples.")
            
            # Conversion au format SpaCy interne : [(text, {"entities": ...}), ...]
            training_data = []
            for item in json_data:
                if 'text' in item and 'entities' in item:
                    training_data.append((item['text'], {"entities": item['entities']}))
            
            if not training_data:
                raise ValueError("Aucune donnée valide trouvée dans le fichier.")
                
            self.generated_training_data = training_data
            self.training_data_path = filepath
            
            self.data_status_label.config(text=f"✅ {len(training_data)} exemples chargés", fg="green")
            self.update_status(f"Données chargées : {len(training_data)} exemples.")
            
            if messagebox.askyesno("Aperçu", "Voulez-vous voir un aperçu des données chargées ?"):
                preview = self.data_generator.preview_training_data(training_data, max_examples=3)
                messagebox.showinfo("Aperçu des Données", preview)
        except Exception as e:
            messagebox.showerror("Erreur de chargement", f"Impossible de charger ou de valider le fichier :\n{e}")

    # --- MÉTHODES D'ENTRAÎNEMENT CORRIGÉES ---

    def set_training_state(self, is_training):
        """ Active ou désactive les contrôles de l'interface pendant l'entraînement. """
        state = 'disabled' if is_training else 'normal'
        
        # Bouton d'entraînement
        self.train_button.config(state=state)
        
        # Contrôles des paramètres
        self.epochs_spinbox.config(state=state)
        self.batch_spinbox.config(state=state)

        # Désactive les autres onglets
        for i, tab in enumerate(self.notebook.tabs()):
            if i != 2: # Ne désactive pas l'onglet d'entraînement lui-même
                self.notebook.tab(i, state=state)

    def start_training(self):
        """ Lance l'entraînement du modèle (version robuste et corrigée). """
        if self.training_in_progress:
            messagebox.showwarning("Entraînement en cours", "Veuillez attendre la fin de l'entraînement actuel.")
            return

        if not self.generated_training_data:
            messagebox.showwarning("Données manquantes", "Veuillez générer ou charger des données d'entraînement.")
            return
            
        if not self.custom_entities:
            messagebox.showwarning("Configuration manquante", "Veuillez définir au moins une entité personnalisée.")
            return
        
        try:
            self.model_trainer = SpacyModelTrainer(self.selected_base_model.get())
            
            if not self.model_trainer.load_base_model():
                messagebox.showerror("Erreur de modèle", f"Impossible de charger le modèle de base : {self.selected_base_model.get()}")
                return
            
            if not self.model_trainer.add_custom_entities(self.custom_entities):
                messagebox.showerror("Erreur de configuration", "Impossible d'ajouter les entités personnalisées au modèle.")
                return
            
            training_config = {
                'n_iter': self.epochs_var.get(),
                'batch_size': self.batch_size_var.get(),
                'dropout': 0.2,
                'patience': 5,
                'validation_split': 0.2
            }
            
            self.training_log.config(state='normal')
            self.training_log.delete(1.0, tk.END)
            self.log_training_message("🚀 Initialisation de l'entraînement...\n")
            self.training_log.config(state='disabled')

            training_thread = threading.Thread(
                target=self._run_training,
                args=(training_config,),
                daemon=True
            )
            training_thread.start()
            
        except Exception as e:
            messagebox.showerror("Erreur de lancement", f"Impossible de démarrer l'entraînement : {e}")

    def _run_training(self, config):
        """ Exécute l'entraînement dans un thread pour ne pas geler l'interface. """
        try:
            self.training_in_progress = True
            self.root.after(0, self.set_training_state, True)

            def progress_callback(current_epoch, total_epochs, epoch_info):
                self.root.after(0, self.progress_var.set, (current_epoch / total_epochs) * 100)
                log_msg = f"Époque {current_epoch}/{total_epochs} | Perte: {epoch_info.get('train_loss', 0):.4f} | F1-Score (Val): {epoch_info.get('val_f1', 0):.3f}\n"
                self.root.after(0, self.log_training_message, log_msg)
            
            results = self.model_trainer.train_model(self.generated_training_data, config, progress_callback)
            self.root.after(0, self._handle_training_results, results)
            
        except Exception as e:
            error_msg = f"❌ Erreur critique pendant l'entraînement: {e}\n"
            self.root.after(0, self.log_training_message, error_msg)
            # Correction de la NameError avec une lambda qui capture 'e'
            self.root.after(0, lambda err=e: messagebox.showerror("Erreur d'entraînement", str(err)))
            
        finally:
            self.training_in_progress = False
            # Le bloc finally garantit que l'interface est toujours réactivée
            self.root.after(0, self.set_training_state, False)

    def _handle_training_results(self, results):
        """ Traite et affiche les résultats à la fin de l'entraînement. """
        self.progress_var.set(0)
        if results.get('success', False):
            final_metrics = results.get('final_metrics', {})
            success_msg = f"🎉 Entraînement terminé !\n📊 F1-Score final: {final_metrics.get('f1', 0):.3f}\n"
            self.log_training_message(success_msg)
            self.update_status(f"✅ Entraînement terminé. F1-Score: {final_metrics.get('f1', 0):.3f}")
            
            if messagebox.askyesno("Sauvegarde du Modèle", f"Entraînement réussi !\nF1-Score: {final_metrics.get('f1', 0):.3f}\n\nVoulez-vous sauvegarder ce modèle ?"):
                self.save_trained_model(results)
        else:
            error_msg = f"❌ Échec de l'entraînement: {results.get('error', 'Erreur inconnue')}\n"
            self.log_training_message(error_msg)
            self.update_status("❌ Échec de l'entraînement.")
            messagebox.showerror("Échec de l'entraînement", results.get('error', 'Erreur inconnue'))

    def save_trained_model(self, training_results):
        """ Sauvegarde le modèle entraîné. """
        save_path = filedialog.askdirectory(title="Choisir un dossier pour sauvegarder le modèle")
        if not save_path:
            return
            
        try:
            model_info = {
                'training_date': datetime.now().isoformat(),
                'base_model': self.selected_base_model.get(),
                'custom_entities': self.custom_entities,
                'final_metrics': training_results.get('final_metrics', {})
            }
            model_path = self.model_trainer.save_model(save_path, model_info)
            self.trained_model_path = model_path
            messagebox.showinfo("Sauvegarde réussie", f"Modèle sauvegardé dans :\n{model_path}")
            self.log_training_message(f"💾 Modèle sauvegardé : {model_path}\n")
        except Exception as e:
            messagebox.showerror("Erreur de sauvegarde", f"Erreur : {e}")

    # --- AUTRES MÉTHODES ---
    
    def log_training_message(self, message):
        """ Ajoute un message au log d'entraînement de manière sécurisée. """
        self.training_log.config(state='normal')
        self.training_log.insert(tk.END, message)
        self.training_log.see(tk.END)
        self.training_log.config(state='disabled')
        self.root.update_idletasks()

    def select_trained_model(self):
        """ Permet de sélectionner un dossier contenant un modèle entraîné. """
        model_path = filedialog.askdirectory(title="Sélectionner le dossier du modèle entraîné")
        if not model_path:
            return

        try:
            # On utilise une instance temporaire pour ne pas écraser le trainer actuel
            test_trainer = SpacyModelTrainer()
            if test_trainer.load_trained_model(model_path):
                self.trained_model_path = model_path
                model_name = Path(model_path).name
                self.model_status_label.config(text=f"✅ Modèle chargé: {model_name}", fg="green")
                
                model_info = test_trainer.get_model_info()
                info_text = (f"Modèle: {model_info.get('base_model', 'N/A')}\n"
                             f"Entités: {', '.join(model_info.get('custom_entities', []))}\n"
                             f"Pipeline: {', '.join(model_info.get('pipeline_components', []))}")
                messagebox.showinfo("Modèle Chargé", info_text)
                self.update_status(f"Modèle sélectionné : {model_name}")
            else:
                messagebox.showerror("Erreur", "Le dossier sélectionné ne semble pas contenir un modèle SpaCy valide.")
        except Exception as e:
            messagebox.showerror("Erreur de chargement", f"Erreur : {e}")

    def test_trained_model(self):
        """ Permet de tester le modèle actuellement chargé. """
        if not self.trained_model_path:
            messagebox.showwarning("Modèle manquant", "Veuillez d'abord sélectionner un modèle entraîné.")
            return
        
        test_dialog = TestModelDialog(self.root)
        if not test_dialog.result:
            return
            
        try:
            if not self.model_trainer or self.model_trainer.nlp is None:
                self.model_trainer = SpacyModelTrainer()
                self.model_trainer.load_trained_model(self.trained_model_path)

            results = self.model_trainer.test_model(test_dialog.result)
            
            if results.get('processed_successfully'):
                entities = results.get('entities', [])
                result_text = f"Texte analysé :\n{test_dialog.result}\n\nEntités détectées ({len(entities)}):\n"
                result_text += "\n".join([f"- '{ent['text']}' ({ent['label']})" for ent in entities])
                if not entities:
                    result_text += "Aucune entité détectée."
                messagebox.showinfo("Résultats du Test", result_text)
            else:
                messagebox.showerror("Erreur de test", results.get('error', 'Erreur inconnue'))
        except Exception as e:
            messagebox.showerror("Erreur de test", f"Erreur : {e}")
    
    def pseudonymize_text(self):
        """ Pseudonymise le texte de l'onglet de pseudonymisation. """
        if not self.trained_model_path:
            messagebox.showwarning("Modèle manquant", "Veuillez sélectionner un modèle entraîné.")
            return
        
        input_text = self.input_text.get(1.0, tk.END).strip()
        if not input_text:
            messagebox.showwarning("Texte manquant", "Veuillez saisir un texte à pseudonymiser.")
            return
        
        try:
            if self.pseudonymizer is None:
                self.pseudonymizer = TextPseudonymizer()
                self.pseudonymizer.load_model(self.trained_model_path)

            # Utilise les entités du modèle chargé pour le dialogue
            entities_in_model = self.pseudonymizer.nlp.get_pipe("ner").labels
            
            entity_selection = EntityMaskingDialog(self.root, list(entities_in_model))
            if entity_selection.result is None: # L'utilisateur a annulé
                return

            pseudonymized_text, stats = self.pseudonymizer.pseudonymize_text(
                input_text,
                entity_types_to_mask=entity_selection.result if entity_selection.result else None
            )
            
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(1.0, pseudonymized_text)
            
            stats_message = self._format_pseudonymization_stats(stats)
            if messagebox.askyesno("Sauvegarde", f"Pseudonymisation terminée !\n\n{stats_message}\n\nVoulez-vous sauvegarder le fichier de correspondance ?"):
                self.save_correspondence_file(stats)
            self.update_status(f"Pseudonymisation terminée : {stats['entities_processed']} entités traitées.")
            
        except Exception as e:
            messagebox.showerror("Erreur de pseudonymisation", f"Erreur : {e}")

    def _format_pseudonymization_stats(self, stats):
        """ Met en forme les statistiques de pseudonymisation pour l'affichage. """
        text = f"Entités traitées: {stats['entities_processed']}\n"
        text += f"Nouveaux pseudonymes: {stats['pseudonyms_created']}\n"
        text += f"Pseudonymes réutilisés: {stats['pseudonyms_reused']}"
        return text

    def save_correspondence_file(self, pseudonymization_stats):
        """ Sauvegarde le fichier de correspondance après pseudonymisation. """
        filepath = filedialog.asksaveasfilename(title="Sauvegarder le fichier de correspondance", defaultextension=".json", filetypes=[("Fichiers JSON", "*.json")])
        if filepath:
            try:
                saved_path = self.pseudonymizer.save_correspondence_file(filepath)
                self.correspondence_file_path = saved_path
                messagebox.showinfo("Sauvegarde réussie", f"Fichier sauvegardé dans:\n{saved_path}")
            except Exception as e:
                messagebox.showerror("Erreur de sauvegarde", f"Erreur: {e}")

    def copy_to_depseudo(self):
        """ Copie le texte pseudonymisé vers l'onglet de dépseudonymisation. """
        pseudonymized_text = self.output_text.get(1.0, tk.END).strip()
        if pseudonymized_text:
            self.pseudo_input_text.delete(1.0, tk.END)
            self.pseudo_input_text.insert(1.0, pseudonymized_text)
            self.notebook.select(4)
            self.update_status("Texte copié pour la dépseudonymisation.")
        else:
            messagebox.showwarning("Aucun texte", "Il n'y a pas de texte à copier.")
    
    def load_correspondence_file(self):
        """ Charge un fichier de correspondance pour la dépseudonymisation. """
        filepath = filedialog.askopenfilename(title="Charger un fichier de correspondance", filetypes=[("Fichiers JSON", "*.json")])
        if not filepath:
            return
            
        try:
            if self.pseudonymizer is None:
                self.pseudonymizer = TextPseudonymizer()

            if self.pseudonymizer.load_correspondence_file(filepath):
                self.correspondence_file_path = filepath
                filename = Path(filepath).name
                self.corresp_status_label.config(text=f"✅ Fichier chargé: {filename}", fg="green")
                summary = self.pseudonymizer.get_pseudonymization_summary()
                self.update_status(f"Correspondances chargées : {summary['total_pseudonyms']} pseudonymes.")
            else:
                messagebox.showerror("Erreur", "Impossible de charger ce fichier de correspondance.")
        except Exception as e:
            messagebox.showerror("Erreur de chargement", f"Erreur : {e}")

    def depseudonymize_text(self):
        """ Dépseudonymise le texte en utilisant le fichier de correspondance chargé. """
        if not self.correspondence_file_path:
            messagebox.showwarning("Fichier manquant", "Veuillez charger un fichier de correspondance.")
            return
        
        pseudo_text = self.pseudo_input_text.get(1.0, tk.END).strip()
        if not pseudo_text:
            messagebox.showwarning("Texte manquant", "Veuillez saisir un texte à dépseudonymiser.")
            return
        
        try:
            depseudonymized_text = self.pseudonymizer.depseudonymize_text(pseudo_text)
            self.depseudo_output_text.delete(1.0, tk.END)
            self.depseudo_output_text.insert(1.0, depseudonymized_text)
            self.update_status("Dépseudonymisation terminée avec succès.")
        except Exception as e:
            messagebox.showerror("Erreur de dépseudonymisation", f"Erreur : {e}")
    
    def import_text_file(self, text_widget):
        """ Utilitaire pour importer un fichier texte dans une zone de texte. """
        filepath = filedialog.askopenfilename(title="Importer un fichier texte", filetypes=[("Fichiers texte", "*.txt")])
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                text_widget.delete(1.0, tk.END)
                text_widget.insert(1.0, content)
                self.update_status(f"Fichier importé : {Path(filepath).name}")
            except Exception as e:
                messagebox.showerror("Erreur d'importation", f"Erreur : {e}")

    def export_text_file(self, text_widget, default_name="exported_text.txt"):
        """ Utilitaire pour exporter le contenu d'une zone de texte. """
        content = text_widget.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("Contenu vide", "Il n'y a rien à exporter.")
            return
        
        filepath = filedialog.asksaveasfilename(title="Exporter le texte", defaultextension=".txt", initialfile=default_name, filetypes=[("Fichiers texte", "*.txt")])
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Export réussi", f"Fichier exporté vers :\n{filepath}")
            except Exception as e:
                messagebox.showerror("Erreur d'exportation", f"Erreur : {e}")
    
    def update_status(self, message):
        """ Met à jour la barre de statut en bas de la fenêtre. """
        self.status_bar.config(text=message)

# ======================
# CLASSES DE DIALOGUES
# ======================

class EntitySelectionDialog:
    """
    Dialogue pour sélectionner un type d'entité
    """
    def __init__(self, parent, entities):
        self.result = None
        
        # Crée la fenêtre de dialogue
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Sélection du type d'entité")
        self.dialog.geometry("300x200")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Centre la fenêtre
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        # Contenu du dialogue
        tk.Label(self.dialog, text="Sélectionnez le type d'entité pour ce fichier:",
                font=("Arial", 10)).pack(pady=10)
        
        self.selected_entity = tk.StringVar(value=entities[0] if entities else "")
        
        for entity in entities:
            rb = tk.Radiobutton(self.dialog, text=entity, 
                               variable=self.selected_entity, value=entity)
            rb.pack(anchor=tk.W, padx=20)
        
        # Boutons
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="OK", command=self.ok_clicked,
                 bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Annuler", command=self.cancel_clicked,
                 bg="#f44336", fg="white").pack(side=tk.LEFT, padx=5)
        
        # Attend la fermeture du dialogue
        self.dialog.wait_window()
    
    def ok_clicked(self):
        self.result = self.selected_entity.get()
        self.dialog.destroy()
    
    def cancel_clicked(self):
        self.result = None
        self.dialog.destroy()


class EntityMaskingDialog:
    """
    Dialogue pour sélectionner les types d'entités à pseudonymiser
    """
    def __init__(self, parent, available_entities):
        self.result = None
        
        # Crée la fenêtre de dialogue
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Sélection des entités à pseudonymiser")
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Centre la fenêtre
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        # Contenu du dialogue
        tk.Label(self.dialog, text="Sélectionnez les types d'entités à pseudonymiser:",
                font=("Arial", 10)).pack(pady=10)
        
        # Frame pour les checkboxes
        checkboxes_frame = tk.Frame(self.dialog)
        checkboxes_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Variables pour les checkboxes
        self.entity_vars = {}
        
        # Option "Toutes les entités"
        self.all_entities_var = tk.BooleanVar(value=True)
        all_cb = tk.Checkbutton(checkboxes_frame, text="Toutes les entités", 
                               variable=self.all_entities_var,
                               command=self.toggle_all_entities,
                               font=("Arial", 10, "bold"))
        all_cb.pack(anchor=tk.W, pady=5)
        
        # Séparateur
        tk.Frame(checkboxes_frame, height=2, bg="gray").pack(fill=tk.X, pady=5)
        
        # Checkboxes pour chaque type d'entité
        for entity in available_entities:
            var = tk.BooleanVar(value=False)
            self.entity_vars[entity] = var
            
            cb = tk.Checkbutton(checkboxes_frame, text=entity, 
                               variable=var,
                               command=self.update_all_checkbox)
            cb.pack(anchor=tk.W, pady=2)
        
        # Boutons
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="OK", command=self.ok_clicked,
                 bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Annuler", command=self.cancel_clicked,
                 bg="#f44336", fg="white").pack(side=tk.LEFT, padx=5)
        
        # Attend la fermeture du dialogue
        self.dialog.wait_window()
    
    def toggle_all_entities(self):
        """Active/désactive toutes les entités"""
        all_selected = self.all_entities_var.get()
        for var in self.entity_vars.values():
            var.set(not all_selected)  # Inverse car "Toutes" signifie ne rien sélectionner spécifiquement
    
    def update_all_checkbox(self):
        """Met à jour la checkbox "Toutes les entités" selon les sélections"""
        any_selected = any(var.get() for var in self.entity_vars.values())
        self.all_entities_var.set(not any_selected)
    
    def ok_clicked(self):
        if self.all_entities_var.get():
            # Toutes les entités sélectionnées
            self.result = []  # Liste vide = toutes les entités
        else:
            # Entités spécifiques sélectionnées
            selected = [entity for entity, var in self.entity_vars.items() if var.get()]
            self.result = selected if selected else []
        
        self.dialog.destroy()
    
    def cancel_clicked(self):
        self.result = None
        self.dialog.destroy()


class TestModelDialog:
    """
    Dialogue pour tester un modèle entraîné
    """
    def __init__(self, parent):
        self.result = None
        
        # Crée la fenêtre de dialogue
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Test du Modèle")
        self.dialog.geometry("500x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Centre la fenêtre
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        # Contenu du dialogue
        tk.Label(self.dialog, text="Saisissez un texte pour tester le modèle:",
                font=("Arial", 10)).pack(pady=10)
        
        # Zone de saisie de texte
        text_frame = tk.Frame(self.dialog)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.text_widget = scrolledtext.ScrolledText(text_frame, height=8, wrap=tk.WORD)
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Texte d'exemple
        example_text = ("Monsieur Dupont travaille à l'établissement ABC123. "
                       "L'organisation XYZ Corp a son siège à Paris. "
                       "Le code d'identification EST-456 correspond à notre filiale.")
        self.text_widget.insert(1.0, example_text)
        
        # Boutons
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="Tester", command=self.test_clicked,
                 bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Annuler", command=self.cancel_clicked,
                 bg="#f44336", fg="white").pack(side=tk.LEFT, padx=5)
        
        # Attend la fermeture du dialogue
        self.dialog.wait_window()
    
    def test_clicked(self):
        self.result = self.text_widget.get(1.0, tk.END).strip()
        self.dialog.destroy()
    
    def cancel_clicked(self):
        self.result = None
        self.dialog.destroy()


class ProgressDialog:
    """
    Dialogue de progression simple
    """
    def __init__(self, parent, message):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Progression")
        self.dialog.geometry("400x100")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Centre la fenêtre
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 100, parent.winfo_rooty() + 100))
        
        tk.Label(self.dialog, text=message, font=("Arial", 10)).pack(pady=10)
        
        # Barre de progression indéterminée
        self.progress = ttk.Progressbar(self.dialog, mode='indeterminate')
        self.progress.pack(pady=10, padx=20, fill=tk.X)
        self.progress.start()
        
        # Met à jour l'affichage
        self.dialog.update()
    
    def destroy(self):
        self.progress.stop()
        self.dialog.destroy()


# ======================
# FONCTION PRINCIPALE
# ======================

def main():
    """
    Fonction principale qui lance l'application
    """
    # Création de la fenêtre principale
    root = tk.Tk()
    
    # Initialisation de l'application
    app = PseudonymizationApp(root)
    
    # Lancement de la boucle principale
    root.mainloop()


if __name__ == "__main__":
    main()


# def start_training(self):
#     """
#     Lance l'entraînement du modèle avec les données chargées
#     """
#     if not self.generated_training_data:
#         messagebox.showwarning("Données manquantes", 
#                              "Veuillez d'abord générer ou charger des données d'entraînement")
#     return
    
#     # Vérifie le format des données avant l'entraînement
#     if not isinstance(self.generated_training_data, list):
#         messagebox.showerror("Erreur de format", 
#                            "Les données d'entraînement ne sont pas au bon format")
#         return
    
#     if len(self.generated_training_data) == 0:
#         messagebox.showerror("Données vides", 
#                            "Aucune donnée d'entraînement disponible")
#         return
    
#     # Vérifie le format du premier élément
#     try:
#         first_item = self.generated_training_data[0]
#         if not (isinstance(first_item, (list, tuple)) and len(first_item) == 2):
#             raise ValueError("Format d'item invalide")
        
#         text, annotations = first_item
#         if not isinstance(text, str) or not isinstance(annotations, dict):
#             raise ValueError("Types d'item invalides")
            
#     except (IndexError, ValueError) as e:
#         messagebox.showerror("Erreur de format", 
#                            f"Format des données d'entraînement invalide: {e}")
#         return
    
#     if not self.custom_entities:
#         messagebox.showwarning("Configuration manquante", 
#                              "Veuillez d'abord configurer vos entités personnalisées")
#         return
    
#     # Vérifie si un entraînement est déjà en cours
#     if hasattr(self, 'training_in_progress') and self.training_in_progress:
#         messagebox.showwarning("Entraînement en cours", 
#                              "Un entraînement est déjà en cours. Veuillez patienter.")
#         return
    
#     try:
#         # Initialise le trainer
#         self.model_trainer = SpacyModelTrainer(self.selected_base_model.get())
        
#         # Charge le modèle de base
#         self.log_training_message("📥 Chargement du modèle de base...\n")
#         if not self.model_trainer.load_base_model():
#             messagebox.showerror("Erreur de modèle", 
#                                f"Impossible de charger le modèle {self.selected_base_model.get()}")
#             return
        
#         # Ajoute les entités personnalisées
#         self.log_training_message("🏷️ Ajout des entités personnalisées...\n")
#         if not self.model_trainer.add_custom_entities(self.custom_entities):
#             messagebox.showerror("Erreur de configuration", 
#                                "Impossible d'ajouter les entités personnalisées")
#             return
        
#         # Configuration d'entraînement
#         training_config = {
#             'n_iter': self.epochs_var.get(),
#             'batch_size': self.batch_size_var.get(),
#             'dropout': 0.2,
#             'patience': 5,
#             'validation_split': 0.2
#         }
        
#         # Efface le log précédent
#         self.training_log.delete(1.0, tk.END)
#         self.log_training_message("🚀 Initialisation de l'entraînement...\n")
#         self.log_training_message(f"📊 Configuration: {training_config}\n")
#         self.log_training_message(f"📦 Données: {len(self.generated_training_data)} exemples\n\n")
        
#         # Désactive les boutons pendant l'entraînement
#         self.set_training_state(True)
        
#         # Lance l'entraînement dans un thread séparé
#         training_thread = threading.Thread(
#             target=self._run_training,
#             args=(training_config,),
#             daemon=True
#         )
#         training_thread.start()
        
#         # Marque l'entraînement comme en cours
#         self.training_in_progress = True
        
#         self.update_status("Entraînement en cours - Veuillez patienter...")
        
#     except Exception as e:
#         self.log_training_message(f"❌ Erreur lors de l'initialisation: {e}\n")
#         messagebox.showerror("Erreur d'entraînement", f"Erreur lors du lancement: {e}")
#         self.set_training_state(False)

# def set_training_state(self, training_active):
#     # """
#     # Active ou désactive les éléments de l'interface pendant l'entraînement
    
#     # Args:
#     #     training_active (bool): True si l'entraînement est en cours
#     # """
#     try:
#         # État des widgets
#         state = 'disabled' if training_active else 'normal'
        

        
#         # Bouton d'entraînement spécifique
#         if hasattr(self, 'train_button'):
#             if training_active:
#                 self.train_button.config(
#                     text="🔄 Entraînement en cours...", 
#                     state='disabled', 
#                     bg="#FF5722"
#                 )
#             else:
#                 self.training_status_label.config(
#                     text="✅ Prêt pour l'entraînement", 
#                     fg="green"
#                 )
        
#         # Désactive les spinbox de paramètres
#         if hasattr(self, 'epochs_var'):
#             # Trouve le spinbox des époques
#             training_frame = self.notebook.nametowidget(self.notebook.tabs()[2])
#             for widget in training_frame.winfo_children():
#                 if isinstance(widget, ttk.LabelFrame) and "Paramètres" in widget.cget('text'):
#                     for child in widget.winfo_children():
#                         if isinstance(child, tk.Frame):
#                             for subchild in child.winfo_children():
#                                 if isinstance(subchild, tk.Spinbox):
#                                     subchild.config(state=state)
        
#         # Met à jour le message de statut
#         if training_active:
#             self.update_status("🔄 Entraînement en cours - Veuillez patienter...")
        
#     except Exception as e:
#         print(f"Erreur lors de la mise à jour de l'état de l'interface: {e}")

# def _run_training(self, config):
#     """
#     Exécute l'entraînement dans un thread séparé
    
#     Args:
#         config: Configuration d'entraînement
#     """
#     try:
#         # Message de début
#         self.root.after(0, lambda: self.log_training_message("⚡ Lancement de l'entraînement...\n"))
        
#         # Fonction de callback pour le suivi de progression
#         def progress_callback(current_epoch, total_epochs, epoch_info):
#             # Met à jour la barre de progression
#             progress_percent = (current_epoch / total_epochs) * 100
#             self.root.after(0, lambda: self.progress_var.set(progress_percent))
            
#             # Met à jour le log
#             log_message = (f"Époque {current_epoch:2d}/{total_epochs} | "
#                           f"Loss: {epoch_info.get('train_loss', 0):.4f} | "
#                           f"Val F1: {epoch_info.get('val_f1', 0):.3f} | "
#                           f"Temps: {epoch_info.get('epoch_time', 0):.1f}s\n")
            
#             self.root.after(0, lambda: self.log_training_message(log_message))
            
#             # Force la mise à jour de l'interface
#             self.root.after(0, lambda: self.root.update_idletasks())
        
#         # Lance l'entraînement
#         self.root.after(0, lambda: self.log_training_message("🎯 Début de l'entraînement proprement dit...\n"))
        
#         results = self.model_trainer.train_model(
#             self.generated_training_data,
#             config,
#             progress_callback
#         )
        
#         # Traite les résultats dans le thread principal
#         self.root.after(0, lambda: self._handle_training_results(results))
        
#     except Exception as e:
#         error_msg = f"❌ Erreur pendant l'entraînement: {e}\n"
#         self.root.after(0, lambda: self.log_training_message(error_msg))
#         self.root.after(0, lambda: messagebox.showerror("Erreur d'entraînement", str(e)))
#         # Réactive l'interface en cas d'erreur
#         self.root.after(0, lambda: self.set_training_state(False))
#         self.root.after(0, lambda: setattr(self, 'training_in_progress', False))

# def _handle_training_results(self, results):
    # """
    # Traite les résultats de l'entraînement
    
    # Args:
    #     results: Résultats retournés par le trainer
    # """
    # # Réactive l'interface
    # self.set_training_state(False)
    # self.training_in_progress = False
    
    # if results.get('success', False):
    #     # Entraînement réussi
    #     final_metrics = results.get('final_metrics', {})
        
    #     success_message = (f"\n🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!\n"
    #                       f"{'='*50}\n"
    #                       f"📊 Métriques finales:\n"
    #                       f"  - Précision: {final_metrics.get('precision', 0):.3f}\n"
    #                       f"  - Rappel: {final_metrics.get('recall', 0):.3f}\n"
    #                       f"  - F1-Score: {final_metrics.get('f1', 0):.3f}\n"
    #                       f"  - Exactitude: {final_metrics.get('accuracy', 0):.3f}\n"
    #                       f"  - Époques complétées: {results.get('epochs_completed', 0)}\n"
    #                       f"{'='*50}\n\n")
        
    #     self.log_training_message(success_message)
        
    #     # Propose de sauvegarder le modèle
    #     if messagebox.askyesno("Sauvegarde du modèle", 
    #                          "🎉 Entraînement terminé avec succès!\n\n"
    #                          f"F1-Score final: {final_metrics.get('f1', 0):.3f}\n"
    #                          f"Précision: {final_metrics.get('precision', 0):.3f}\n"
    #                          f"Rappel: {final_metrics.get('recall', 0):.3f}\n\n"
    #                          "Voulez-vous sauvegarder le modèle entraîné ?"):
    #         self.save_trained_model(results)
        
    #     # Met à jour le statut
    #     self.update_status(f"✅ Entraînement terminé - F1-Score: {final_metrics.get('f1', 0):.3f}")
        
    # else:
    #     # Entraînement échoué
    #     error_message = f"\n❌ ÉCHEC DE L'ENTRAÎNEMENT\n{'='*30}\n{results.get('error', 'Erreur inconnue')}\n\n"
    #     self.log_training_message(error_message)
    #     messagebox.showerror("Échec de l'entraînement", 
    #                        f"L'entraînement a échoué:\n\n{results.get('error', 'Erreur inconnue')}")
    #     self.update_status("❌ Échec de l'entraînement")
    
    # # Remet la barre de progression à zéro
    # self.progress_var.set(0)
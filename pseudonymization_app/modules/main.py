#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application de Pseudonymisation avec SpaCy
==========================================

Cette application permet de :
1. Entraîner des modèles SpaCy personnalisés avec de nouvelles entités NER
2. Générer automatiquement des données d'entraînement
3. Pseudonymiser et dépseudonymiser des textes

Auteur: Assistant IA
Date: 2024
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import json
from pathlib import Path

# Ajout du dossier modules au chemin Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Import des modules personnalisés
from data_generator import TrainingDataGenerator
from model_trainer import SpacyModelTrainer
from pseudonymizer import TextPseudonymizer
from utils import AppUtils

class PseudonymizationApp:
    # """
    # Application principale de pseudonymisation
    
    # Cette classe gère l'interface graphique complète et coordonne
    # tous les modules de l'application pour offrir une expérience
    # utilisateur fluide et intuitive.
    # """
    
    def __init__(self, root):
        # """
        # Initialise l'application principale
        
        # Args:
        #     root: Fenêtre principale Tkinter
        # """
        self.root = root
        self.root.title("Application de Pseudonymisation - SpaCy NER")
        self.root.geometry("1200x800")
        
        # Variables de l'application
        self.selected_base_model = tk.StringVar(value="fr_core_news_sm")
        self.custom_entities = []  # Liste des entités personnalisées
        self.training_data_path = ""
        self.trained_model_path = ""
        
        # Initialisation des modules
        self.data_generator = TrainingDataGenerator()
        self.model_trainer = None
        self.pseudonymizer = None
        self.utils = AppUtils()
        
        # Configuration de l'interface
        self.setup_ui()
        
        # Création des dossiers nécessaires
        self.create_directories()
        # Variables supplémentaires pour la génération de données
        self.entity_files = {}  # Dictionnaire {type_entité: chemin_fichier}
        self.generated_training_data = None
        
        
    def create_directories(self):
        """
        Crée les dossiers nécessaires à l'application s'ils n'existent pas
        """
        directories = ['models', 'data', 'config']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            
    def setup_ui(self):
        """
        Configure l'interface utilisateur avec tous les onglets
        
        Cette méthode crée l'interface complète avec :
        - Onglet de configuration
        - Onglet de génération de données
        - Onglet d'entraînement
        - Onglet de pseudonymisation
        - Onglet de dépseudonymisation
        """
        # Création du notebook (onglets)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Création des onglets
        self.create_config_tab()
        self.create_data_generation_tab()
        self.create_training_tab()
        self.create_pseudonymization_tab()
        self.create_depseudonymization_tab()
        
        # Barre de statut
        self.status_bar = tk.Label(self.root, text="Prêt", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_config_tab(self):
        """
        Crée l'onglet de configuration du modèle de base
        
        Cet onglet permet à l'utilisateur de :
        - Sélectionner un modèle SpaCy de base
        - Définir de nouvelles entités NER personnalisées
        - Valider sa configuration
        """
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="1. Configuration")
        
        # Titre de la section
        title_label = tk.Label(config_frame, text="Configuration du Modèle de Base", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Section sélection du modèle
        model_frame = ttk.LabelFrame(config_frame, text="Modèle SpaCy de Base")
        model_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Liste des modèles disponibles
        models = ["fr_core_news_sm", "fr_core_news_md", "fr_core_news_lg", 
                 "en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
        
        for model in models:
            rb = tk.Radiobutton(model_frame, text=model, variable=self.selected_base_model,
                               value=model, font=("Arial", 10))
            rb.pack(anchor=tk.W, padx=10, pady=2)
        
        # Section entités personnalisées
        entities_frame = ttk.LabelFrame(config_frame, text="Entités NER Personnalisées")
        entities_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Zone de saisie pour nouvelle entité
        add_entity_frame = tk.Frame(entities_frame)
        add_entity_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(add_entity_frame, text="Nom de l'entité:").pack(side=tk.LEFT)
        self.entity_entry = tk.Entry(add_entity_frame, width=30)
        self.entity_entry.pack(side=tk.LEFT, padx=5)
        
        add_button = tk.Button(add_entity_frame, text="Ajouter Entité",
                              command=self.add_custom_entity, bg="#4CAF50", fg="white")
        add_button.pack(side=tk.LEFT, padx=5)
        
        # Liste des entités ajoutées
        self.entities_listbox = tk.Listbox(entities_frame, height=8)
        self.entities_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Bouton de suppression
        remove_button = tk.Button(entities_frame, text="Supprimer Entité Sélectionnée",
                                 command=self.remove_custom_entity, bg="#f44336", fg="white")
        remove_button.pack(pady=5)
        
        # Bouton de validation
        validate_button = tk.Button(config_frame, text="Valider Configuration",
                                   command=self.validate_configuration, 
                                   bg="#2196F3", fg="white", font=("Arial", 12))
        validate_button.pack(pady=20)
        
    def create_data_generation_tab(self):
        """
        Crée l'onglet de génération automatique des données d'entraînement
        
        Permet à l'utilisateur de :
        - Importer des listes de termes pour chaque entité
        - Configurer la génération de phrases
        - Prévisualiser les données générées
        - Exporter les données d'entraînement
        """
        data_gen_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_gen_frame, text="2. Génération de Données")
        
        # Titre
        title_label = tk.Label(data_gen_frame, text="Génération Automatique des Données d'Entraînement", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Section import des fichiers
        import_frame = ttk.LabelFrame(data_gen_frame, text="Import des Listes de Termes")
        import_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Instructions
        instructions = tk.Label(import_frame, 
                               text="Importez un fichier texte (.txt) contenant une liste de termes par ligne pour chaque entité NER",
                               wraplength=800, justify=tk.LEFT)
        instructions.pack(padx=10, pady=5)
        
        # Zone d'import des fichiers
        self.files_frame = tk.Frame(import_frame)
        self.files_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Bouton d'ajout de fichier
        add_file_button = tk.Button(self.files_frame, text="Ajouter Fichier de Termes",
                                   command=self.add_terms_file, bg="#4CAF50", fg="white")
        add_file_button.pack(pady=5)
        
        # Liste des fichiers importés
        self.imported_files_list = tk.Listbox(import_frame, height=5)
        self.imported_files_list.pack(fill=tk.X, padx=10, pady=5)
        
        # Paramètres de génération
        params_frame = ttk.LabelFrame(data_gen_frame, text="Paramètres de Génération")
        params_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Nombre de phrases par entité
        sentences_frame = tk.Frame(params_frame)
        sentences_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(sentences_frame, text="Nombre de phrases par terme:").pack(side=tk.LEFT)
        self.sentences_per_term = tk.IntVar(value=5)
        sentences_spinbox = tk.Spinbox(sentences_frame, from_=1, to=50, 
                                      textvariable=self.sentences_per_term, width=10)
        sentences_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Bouton de génération
        generate_button = tk.Button(data_gen_frame, text="Générer Données d'Entraînement",
                                   command=self.generate_training_data,
                                   bg="#FF9800", fg="white", font=("Arial", 12))
        generate_button.pack(pady=20)
        
        # Zone de prévisualisation
        preview_frame = ttk.LabelFrame(data_gen_frame, text="Aperçu des Données Générées")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.preview_text = scrolledtext.ScrolledText(preview_frame, height=10, wrap=tk.WORD)
        self.preview_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_training_tab(self):
        """
        Crée l'onglet d'entraînement du modèle
        
        Interface pour :
        - Charger les données d'entraînement
        - Configurer les paramètres d'entraînement
        - Suivre le progrès en temps réel
        - Visualiser les métriques
        """
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="3. Entraînement")
        
        # Titre
        title_label = tk.Label(training_frame, text="Fine-tuning du Modèle SpaCy", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Section chargement des données
        data_frame = ttk.LabelFrame(training_frame, text="Données d'Entraînement")
        data_frame.pack(fill=tk.X, padx=20, pady=10)
        
        load_data_button = tk.Button(data_frame, text="Charger Données d'Entraînement",
                                    command=self.load_training_data, bg="#4CAF50", fg="white")
        load_data_button.pack(pady=10)
        
        self.data_status_label = tk.Label(data_frame, text="Aucune donnée chargée", fg="red")
        self.data_status_label.pack()
        
        # Paramètres d'entraînement
        params_frame = ttk.LabelFrame(training_frame, text="Paramètres d'Entraînement")
        params_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Nombre d'époques
        epochs_frame = tk.Frame(params_frame)
        epochs_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(epochs_frame, text="Nombre d'époques:").pack(side=tk.LEFT)
        self.epochs_var = tk.IntVar(value=30)
        epochs_spinbox = tk.Spinbox(epochs_frame, from_=5, to=200, 
                                   textvariable=self.epochs_var, width=10)
        epochs_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Taille de batch
        batch_frame = tk.Frame(params_frame)
        batch_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(batch_frame, text="Taille de batch:").pack(side=tk.LEFT)
        self.batch_size_var = tk.IntVar(value=8)
        batch_spinbox = tk.Spinbox(batch_frame, from_=1, to=32, 
                                  textvariable=self.batch_size_var, width=10)
        batch_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Bouton d'entraînement
        train_button = tk.Button(training_frame, text="Commencer l'Entraînement",
                                command=self.start_training,
                                bg="#2196F3", fg="white", font=("Arial", 12))
        train_button.pack(pady=20)
        
        # Barre de progression
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(training_frame, variable=self.progress_var,
                                           maximum=100, length=600)
        self.progress_bar.pack(pady=10)
        
        # Zone de log d'entraînement
        log_frame = ttk.LabelFrame(training_frame, text="Journal d'Entraînement")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.training_log = scrolledtext.ScrolledText(log_frame, height=8, wrap=tk.WORD)
        self.training_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_pseudonymization_tab(self):
        """
        Crée l'onglet de pseudonymisation des textes
        """
        pseudo_frame = ttk.Frame(self.notebook)
        self.notebook.add(pseudo_frame, text="4. Pseudonymisation")
        
        # Titre
        title_label = tk.Label(pseudo_frame, text="Pseudonymisation de Texte", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Sélection du modèle
        model_frame = ttk.LabelFrame(pseudo_frame, text="Sélection du Modèle")
        model_frame.pack(fill=tk.X, padx=20, pady=10)
        
        select_model_button = tk.Button(model_frame, text="Sélectionner Modèle Entraîné",
                                       command=self.select_trained_model, bg="#4CAF50", fg="white")
        select_model_button.pack(pady=10)
        
        self.model_status_label = tk.Label(model_frame, text="Aucun modèle sélectionné", fg="red")
        self.model_status_label.pack()
        
        # Zone de texte à pseudonymiser
        input_frame = ttk.LabelFrame(pseudo_frame, text="Texte à Pseudonymiser")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.input_text = scrolledtext.ScrolledText(input_frame, height=8, wrap=tk.WORD)
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bouton de pseudonymisation
        pseudo_button = tk.Button(pseudo_frame, text="Pseudonymiser",
                                 command=self.pseudonymize_text,
                                 bg="#FF9800", fg="white", font=("Arial", 12))
        pseudo_button.pack(pady=10)
        
        # Zone de résultat
        output_frame = ttk.LabelFrame(pseudo_frame, text="Texte Pseudonymisé")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=8, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_depseudonymization_tab(self):
        """
        Crée l'onglet de dépseudonymisation
        """
        depseudo_frame = ttk.Frame(self.notebook)
        self.notebook.add(depseudo_frame, text="5. Dépseudonymisation")
        
        # Titre
        title_label = tk.Label(depseudo_frame, text="Dépseudonymisation de Texte", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Chargement du fichier de correspondance
        corresp_frame = ttk.LabelFrame(depseudo_frame, text="Fichier de Correspondance")
        corresp_frame.pack(fill=tk.X, padx=20, pady=10)
        
        load_corresp_button = tk.Button(corresp_frame, text="Charger Fichier de Correspondance",
                                       command=self.load_correspondence_file, bg="#4CAF50", fg="white")
        load_corresp_button.pack(pady=10)
        
        self.corresp_status_label = tk.Label(corresp_frame, text="Aucun fichier chargé", fg="red")
        self.corresp_status_label.pack()
        
        # Zone de texte pseudonymisé
        pseudo_input_frame = ttk.LabelFrame(depseudo_frame, text="Texte Pseudonymisé")
        pseudo_input_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.pseudo_input_text = scrolledtext.ScrolledText(pseudo_input_frame, height=8, wrap=tk.WORD)
        self.pseudo_input_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bouton de dépseudonymisation
        depseudo_button = tk.Button(depseudo_frame, text="Dépseudonymiser",
                                   command=self.depseudonymize_text,
                                   bg="#9C27B0", fg="white", font=("Arial", 12))
        depseudo_button.pack(pady=10)
        
        # Zone de résultat
        depseudo_output_frame = ttk.LabelFrame(depseudo_frame, text="Texte Original Restauré")
        depseudo_output_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.depseudo_output_text = scrolledtext.ScrolledText(depseudo_output_frame, height=8, wrap=tk.WORD)
        self.depseudo_output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Méthodes de gestion des événements (à implémenter dans les prochaines parties)
    def add_custom_entity(self):
        """Ajoute une entité personnalisée à la liste"""
        entity = self.entity_entry.get().strip().upper()
        if entity and entity not in self.custom_entities:
            self.custom_entities.append(entity)
            self.entities_listbox.insert(tk.END, entity)
            self.entity_entry.delete(0, tk.END)
            self.update_status(f"Entité '{entity}' ajoutée")
        elif entity in self.custom_entities:
            messagebox.showwarning("Attention", f"L'entité '{entity}' existe déjà")
    
    def remove_custom_entity(self):
        """Supprime l'entité sélectionnée"""
        selection = self.entities_listbox.curselection()
        if selection:
            entity = self.entities_listbox.get(selection[0])
            self.entities_listbox.delete(selection[0])
            self.custom_entities.remove(entity)
            self.update_status(f"Entité '{entity}' supprimée")
    
    def validate_configuration(self):
        """Valide la configuration du modèle"""
        if not self.custom_entities:
            messagebox.showwarning("Configuration incomplète", 
                                 "Veuillez ajouter au moins une entité personnalisée")
            return
        
        messagebox.showinfo("Configuration validée", 
                           f"Modèle de base: {self.selected_base_model.get()}\n"
                           f"Entités personnalisées: {', '.join(self.custom_entities)}")
        self.update_status("Configuration validée - Passez à l'étape suivante")
    
    def update_status(self, message):
        """Met à jour la barre de statut"""
        self.status_bar.config(text=message)
    
    # Méthodes placeholder - à implémenter dans les prochaines parties
    def add_terms_file(self):
        """
        Opens a dialog for the user to add a terms file for a specific custom entity type.
        - Checks if custom entities are configured; if not, shows a warning and exits.
        - Prompts the user to select a text file containing terms.
        - Prompts the user to select the entity type to associate with the file.
        - Associates the selected file with the chosen entity and updates the UI list.
        - Updates the status bar with a confirmation message.
        - Previews the first few terms from the file in an info dialog.
        - Handles and displays errors if the file cannot be read.
        Requires:
            - self.custom_entities: List of configured custom entity types.
            - self.entity_files: Dictionary mapping entity types to file paths.
            - self.imported_files_list: Tkinter Listbox for displaying imported files.
            - self.update_status: Method to update the status bar.
            - self.data_generator.load_terms_from_file: Method to load terms from a file.
            - EntitySelectionDialog: Dialog class for selecting an entity type.
        """
        # """
        # Permet à l'utilisateur d'ajouter un fichier contenant des termes
        # pour un type d'entité spécifique
        # """
        if not self.custom_entities:
            messagebox.showwarning("Configuration requise", 
                                 "Veuillez d'abord configurer vos entités personnalisées dans l'onglet Configuration")
            return
        
        # Dialogue de sélection du fichier
        filepath = filedialog.askopenfilename(
            title="Sélectionner un fichier de termes",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")]
        )
        
        if not filepath:
            return
        
        # Dialogue pour choisir le type d'entité
        entity_dialog = EntitySelectionDialog(self.root, self.custom_entities)
        selected_entity = entity_dialog.result
        
        if selected_entity:
            # Ajoute le fichier à la liste
            self.entity_files[selected_entity] = filepath
            
            # Met à jour l'affichage
            display_text = f"{selected_entity}: {Path(filepath).name}"
            self.imported_files_list.insert(tk.END, display_text)
            
            self.update_status(f"Fichier ajouté pour l'entité {selected_entity}")
            
            # Prévisualise quelques termes du fichier
            try:
                preview_terms = self.data_generator.load_terms_from_file(filepath)[:5]
                preview_text = ", ".join(preview_terms)
                if len(preview_terms) < 5:
                    messagebox.showinfo("Fichier chargé", 
                                      f"Fichier chargé pour {selected_entity}\n"
                                      f"Termes: {preview_text}")
                else:
                    messagebox.showinfo("Fichier chargé", 
                                      f"Fichier chargé pour {selected_entity}\n"
                                      f"Premiers termes: {preview_text}...")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de la lecture du fichier: {e}")
        
    def generate_training_data(self):
        # """
        # Lance la génération automatique des données d'entraînement
        # """
        if not self.entity_files:
            messagebox.showwarning("Fichiers manquants", 
                                 "Veuillez d'abord ajouter des fichiers de termes")
            return
        
        try:
            # Affiche un dialogue de progression
            progress_dialog = ProgressDialog(self.root, "Génération des données d'entraînement...")
            
            # Génère les données
            self.generated_training_data, stats = self.data_generator.generate_training_data(
                self.entity_files,
                sentences_per_term=self.sentences_per_term.get(),
                add_variations=True
            )
            
            progress_dialog.destroy()
            
            if self.generated_training_data:
                # Affiche un aperçu des données générées
                preview_text = self.data_generator.preview_training_data(
                    self.generated_training_data, max_examples=5
                )
                self.preview_text.delete(1.0, tk.END)
                self.preview_text.insert(1.0, preview_text)
                
                # Propose de sauvegarder
                if messagebox.askyesno("Sauvegarde", 
                                     f"{len(self.generated_training_data)} exemples générés.\n"
                                     "Voulez-vous sauvegarder ces données ?"):
                    try:
                        saved_path = self.data_generator.save_training_data(self.generated_training_data)
                        self.training_data_path = saved_path
                        messagebox.showinfo("Sauvegarde réussie", 
                                          f"Données sauvegardées dans:\n{saved_path}")
                    except Exception as e:
                        messagebox.showerror("Erreur de sauvegarde", f"Erreur: {e}")
                
                self.update_status(f"Génération terminée: {len(self.generated_training_data)} exemples")
            else:
                messagebox.showwarning("Génération échouée", 
                                     "Aucune donnée n'a pu être générée")
        
        except Exception as e:
            messagebox.showerror("Erreur de génération", f"Erreur lors de la génération: {e}")
        
    def load_training_data(self):
        # """
        # Charge des données d'entraînement depuis un fichier JSON
        # """
        filepath = filedialog.askopenfilename(
            title="Sélectionner le fichier de données d'entraînement",
            filetypes=[("Fichiers JSON", "*.json"), ("Tous les fichiers", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Convertit le format JSON vers le format SpaCy
                training_data = []
                for item in json_data:
                    text = item.get('text', '')
                    entities = item.get('entities', [])
                    training_data.append((text, {"entities": entities}))
                
                self.generated_training_data = training_data
                self.training_data_path = filepath
                
                self.data_status_label.config(
                    text=f"✅ {len(training_data)} exemples chargés", 
                    fg="green"
                )
                
                self.update_status(f"Données d'entraînement chargées: {len(training_data)} exemples")
                
            except Exception as e:
                messagebox.showerror("Erreur de chargement", 
                                   f"Impossible de charger le fichier:\n{e}")
        
    def start_training(self):
        # """Commencer l'entraînement"""
        # """
        # Lance l'entraînement du modèle avec les données chargées
        # """
        if not self.generated_training_data:
            messagebox.showwarning("Données manquantes", 
                                 "Veuillez d'abord générer ou charger des données d'entraînement")
            return
        
        if not self.custom_entities:
            messagebox.showwarning("Configuration manquante", 
                                 "Veuillez d'abord configurer vos entités personnalisées")
            return
        
        try:
            # Initialise le trainer
            self.model_trainer = SpacyModelTrainer(self.selected_base_model.get())
            
            # Charge le modèle de base
            if not self.model_trainer.load_base_model():
                messagebox.showerror("Erreur de modèle", 
                                   f"Impossible de charger le modèle {self.selected_base_model.get()}")
                return
            
            # Ajoute les entités personnalisées
            if not self.model_trainer.add_custom_entities(self.custom_entities):
                messagebox.showerror("Erreur de configuration", 
                                   "Impossible d'ajouter les entités personnalisées")
                return
            
            # Configuration d'entraînement
            training_config = {
                'n_iter': self.epochs_var.get(),
                'batch_size': self.batch_size_var.get(),
                'dropout': 0.2,
                'patience': 5,
                'validation_split': 0.2
            }
            
            # Efface le log précédent
            self.training_log.delete(1.0, tk.END)
            self.log_training_message("🚀 Début de l'entraînement...\n")
            
            # Lance l'entraînement dans un thread séparé pour ne pas bloquer l'interface
            import threading
            training_thread = threading.Thread(
                target=self._run_training,
                args=(training_config,)
            )
            training_thread.daemon = True
            training_thread.start()
            
        except Exception as e:
            messagebox.showerror("Erreur d'entraînement", f"Erreur lors du lancement: {e}")

    def _handle_training_results(self, results):
        # """
        # Traite les résultats de l'entraînement
        
        # Args:
        #     results: Résultats retournés par le trainer
        # """
        if results.get('success', False):
            # Entraînement réussi
            final_metrics = results.get('final_metrics', {})
            
            success_message = (f"🎉 Entraînement terminé avec succès!\n"
                              f"📊 Métriques finales:\n"
                              f"  - Précision: {final_metrics.get('precision', 0):.3f}\n"
                              f"  - Rappel: {final_metrics.get('recall', 0):.3f}\n"
                              f"  - F1-Score: {final_metrics.get('f1', 0):.3f}\n"
                              f"  - Époques: {results.get('epochs_completed', 0)}\n\n")
            
            self.log_training_message(success_message)
            
            # Propose de sauvegarder le modèle
            if messagebox.askyesno("Sauvegarde du modèle", 
                                 "Entraînement terminé avec succès!\n"
                                 "Voulez-vous sauvegarder le modèle entraîné ?"):
                self.save_trained_model(results)
            
            # Met à jour le statut
            self.update_status(f"Entraînement terminé - F1-Score: {final_metrics.get('f1', 0):.3f}")
            
        else:
            # Entraînement échoué
            error_message = f"❌ Échec de l'entraînement: {results.get('error', 'Erreur inconnue')}\n"
            self.log_training_message(error_message)
            messagebox.showerror("Échec de l'entraînement", results.get('error', 'Erreur inconnue'))
        
        # Remet la barre de progression à zéro
        self.progress_var.set(0)

def save_trained_model(self, training_results):
    """
    Sauvegarde le modèle entraîné
    
    Args:
        training_results: Résultats de l'entraînement
    """
    try:
        # Dialogue pour choisir l'emplacement
        save_path = filedialog.askdirectory(
            title="Choisir le dossier de sauvegarde du modèle"
        )
        
        if save_path:
            # Informations du modèle
            model_info = {
                'training_date': datetime.now().isoformat(),
                'base_model': self.selected_base_model.get(),
                'custom_entities': self.custom_entities,
                'training_examples': len(self.generated_training_data),
                'final_metrics': training_results.get('final_metrics', {})
            }
            
            # Sauvegarde
            model_path = self.model_trainer.save_model(save_path, model_info)
            self.trained_model_path = model_path
            
            messagebox.showinfo("Sauvegarde réussie", 
                              f"Modèle sauvegardé avec succès dans:\n{model_path}")
            
            self.log_training_message(f"💾 Modèle sauvegardé: {model_path}\n")
            
    except Exception as e:
        messagebox.showerror("Erreur de sauvegarde", f"Erreur lors de la sauvegarde: {e}")

def log_training_message(self, message):
    # """
    # Ajoute un message au log d'entraînement
    
    # Args:
    #     message: Message à ajouter
    # """
    self.training_log.insert(tk.END, message)
    self.training_log.see(tk.END)  # Fait défiler vers le bas
    self.root.update_idletasks()  # Met à jour l'affichage

def select_trained_model(self):
    """
    Permet à l'utilisateur de sélectionner un modèle entraîné
    """
    model_path = filedialog.askdirectory(
        title="Sélectionner le dossier du modèle entraîné"
    )
    
    if model_path:
        try:
            # Teste le chargement du modèle
            test_trainer = SpacyModelTrainer()
            if test_trainer.load_trained_model(model_path):
                self.trained_model_path = model_path
                
                # Met à jour l'affichage
                model_name = Path(model_path).name
                self.model_status_label.config(
                    text=f"✅ Modèle chargé: {model_name}", 
                    fg="green"
                )
                
                # Affiche les informations du modèle
                model_info = test_trainer.get_model_info()
                info_text = (f"Modèle: {model_info.get('base_model', 'Unknown')}\n"
                            f"Entités: {', '.join(model_info.get('custom_entities', []))}\n"
                            f"Composants: {', '.join(model_info.get('pipeline_components', []))}")
                
                messagebox.showinfo("Modèle chargé", info_text)
                
                self.update_status(f"Modèle sélectionné: {model_name}")
                
            else:
                messagebox.showerror("Erreur", "Impossible de charger le modèle sélectionné")
                
        except Exception as e:
            messagebox.showerror("Erreur de chargement", f"Erreur: {e}")

# Ajoutez aussi cette méthode pour tester le modèle
def test_trained_model(self):
    """
    Permet de tester le modèle entraîné sur un texte d'exemple
    """
    if not self.trained_model_path:
        messagebox.showwarning("Modèle manquant", "Veuillez d'abord sélectionner un modèle entraîné")
        return
    
    # Dialogue pour saisir le texte de test
    test_dialog = TestModelDialog(self.root)
    test_text = test_dialog.result
    
    if test_text:
        try:
            # Charge le modèle si nécessaire
            if not hasattr(self, 'model_trainer') or not self.model_trainer:
                self.model_trainer = SpacyModelTrainer()
                self.model_trainer.load_trained_model(self.trained_model_path)
            
            # Teste le modèle
            results = self.model_trainer.test_model(test_text)
            
            if results.get('processed_successfully', False):
                # Affiche les résultats
                entities = results.get('entities', [])
                result_text = f"Texte testé: {test_text}\n\n"
                result_text += f"Entités détectées ({len(entities)}):\n"
                
                for ent in entities:
                    result_text += f"- '{ent['text']}' ({ent['label']}) [pos: {ent['start']}-{ent['end']}]\n"
                
                if not entities:
                    result_text += "Aucune entité détectée.\n"
                
                messagebox.showinfo("Résultats du test", result_text)
            else:
                messagebox.showerror("Erreur de test", results.get('error', 'Erreur inconnue'))
                
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du test: {e}")


    def _run_training(self, config):
        # """
        # Exécute l'entraînement dans un thread séparé
        
        # Args:
        #     config: Configuration d'entraînement
        # """
        try:
            # Fonction de callback pour le suivi de progression
            def progress_callback(current_epoch, total_epochs, epoch_info):
                # Met à jour la barre de progression
                progress_percent = (current_epoch / total_epochs) * 100
                self.root.after(0, lambda: self.progress_var.set(progress_percent))
                
                # Met à jour le log
                log_message = (f"Époque {current_epoch}/{total_epochs} | "
                              f"Loss: {epoch_info.get('train_loss', 0):.4f} | "
                              f"Val F1: {epoch_info.get('val_f1', 0):.3f} | "
                              f"Temps: {epoch_info.get('epoch_time', 0):.1f}s\n")
                
                self.root.after(0, lambda: self.log_training_message(log_message))
            
            # Lance l'entraînement
            results = self.model_trainer.train_model(
                self.generated_training_data,
                config,
                progress_callback
            )
            
            # Traite les résultats dans le thread principal
            self.root.after(0, lambda: self._handle_training_results(results))
        
        except Exception as e:
            error_msg = f"❌ Erreur pendant l'entraînement: {e}\n"
            self.root.after(0, lambda: self.log_training_message(error_msg))
            self.root.after(0, lambda: messagebox.showerror("Erreur d'entraînement", str(e)))
        
    def select_trained_model(self):
        """Sélectionner un modèle entraîné"""
        pass
        
    def pseudonymize_text(self):
        """Pseudonymiser le texte"""
        pass
        
    def load_correspondence_file(self):
        """Charger le fichier de correspondance"""
        pass
        
    def depseudonymize_text(self):
        """Dépseudonymiser le texte"""
        pass

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

# Ajoutez ces classes à la fin de main.py

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
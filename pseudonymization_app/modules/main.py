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
        
        Args:
            root: Fenêtre principale Tkinter
        """
        self.root = root
        self.root.title("Application de Pseudonymisation - SpaCy NER")
        self.root.geometry("1200x800")
        
        # Variables de l'application
        self.selected_base_model = tk.StringVar(value="fr_core_news_sm")
        self.custom_entities = []  # Liste des entités personnalisées
        self.training_data_path = ""
        self.trained_model_path = ""
        
        # Variables pour la génération de données
        self.entity_files = {}  # Dictionnaire {type_entité: chemin_fichier}
        self.generated_training_data = None
        
        # Variables pour la pseudonymisation
        self.pseudonymizer = None
        self.correspondence_file_path = ""
        
        # Initialisation des modules
        self.data_generator = TrainingDataGenerator()
        self.model_trainer = None
        self.utils = AppUtils()
        
        # Configuration de l'interface
        self.setup_ui()
        
        # Création des dossiers nécessaires
        self.create_directories()
        
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
        
        data_buttons_frame = tk.Frame(data_frame)
        data_buttons_frame.pack(pady=10)
        
        load_data_button = tk.Button(data_buttons_frame, text="Charger Données d'Entraînement",
                                    command=self.load_training_data, bg="#4CAF50", fg="white")
        load_data_button.pack(side=tk.LEFT, padx=5)
        
        test_model_button = tk.Button(data_buttons_frame, text="Tester Modèle",
                                     command=self.test_trained_model, bg="#2196F3", fg="white")
        test_model_button.pack(side=tk.LEFT, padx=5)
        
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
        
        # Zone de texte à pseudonymiser avec boutons
        input_frame = ttk.LabelFrame(pseudo_frame, text="Texte à Pseudonymiser")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Boutons d'import/export pour le texte d'entrée
        input_buttons_frame = tk.Frame(input_frame)
        input_buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(input_buttons_frame, text="Importer Fichier", 
                 command=lambda: self.import_text_file(self.input_text),
                 bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(input_buttons_frame, text="Effacer", 
                 command=lambda: self.input_text.delete(1.0, tk.END),
                 bg="#FF5722", fg="white").pack(side=tk.LEFT, padx=5)
        
        self.input_text = scrolledtext.ScrolledText(input_frame, height=8, wrap=tk.WORD)
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bouton de pseudonymisation
        pseudo_button = tk.Button(pseudo_frame, text="Pseudonymiser",
                                 command=self.pseudonymize_text,
                                 bg="#FF9800", fg="white", font=("Arial", 12))
        pseudo_button.pack(pady=10)
        
        # Zone de résultat avec boutons
        output_frame = ttk.LabelFrame(pseudo_frame, text="Texte Pseudonymisé")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Boutons d'export pour le texte de sortie
        output_buttons_frame = tk.Frame(output_frame)
        output_buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(output_buttons_frame, text="Exporter Fichier", 
                 command=lambda: self.export_text_file(self.output_text, "texte_pseudonymise.txt"),
                 bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(output_buttons_frame, text="Copier vers Dépseudonymisation", 
                 command=self.copy_to_depseudo,
                 bg="#9C27B0", fg="white").pack(side=tk.LEFT, padx=5)
        
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
        
        # Boutons pour le texte pseudonymisé
        pseudo_buttons_frame = tk.Frame(pseudo_input_frame)
        pseudo_buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(pseudo_buttons_frame, text="Importer Fichier", 
                 command=lambda: self.import_text_file(self.pseudo_input_text),
                 bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(pseudo_buttons_frame, text="Effacer", 
                 command=lambda: self.pseudo_input_text.delete(1.0, tk.END),
                 bg="#FF5722", fg="white").pack(side=tk.LEFT, padx=5)
        
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
        
        # Boutons pour le texte restauré
        restore_buttons_frame = tk.Frame(depseudo_output_frame)
        restore_buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(restore_buttons_frame, text="Exporter Fichier", 
                 command=lambda: self.export_text_file(self.depseudo_output_text, "texte_original.txt"),
                 bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        
        self.depseudo_output_text = scrolledtext.ScrolledText(depseudo_output_frame, height=8, wrap=tk.WORD)
        self.depseudo_output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # ======================
    # MÉTHODES DE CONFIGURATION
    # ======================
    
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
    
    # ======================
    # MÉTHODES DE GÉNÉRATION DE DONNÉES
    # ======================
    
    def add_terms_file(self):
        """
        Permet à l'utilisateur d'ajouter un fichier contenant des termes
        pour un type d'entité spécifique
        """
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
        """
        Lance la génération automatique des données d'entraînement
        """
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
        """
        Charge des données d'entraînement depuis un fichier JSON
        """
        filepath = filedialog.askopenfilename(
            title="Sélectionner le fichier de données d'entraînement",
            filetypes=[("Fichiers JSON", "*.json"), ("Tous les fichiers", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Vérifie le format des données
                if not json_data:
                    raise ValueError("Le fichier est vide")
                
                # Convertit le format JSON vers le format SpaCy
                training_data = []
                
                # Vérifie si c'est déjà une liste de tuples (données générées directement)
                if isinstance(json_data, list) and len(json_data) > 0:
                    first_item = json_data[0]
                    
                    # Si c'est déjà au format SpaCy (liste de tuples)
                    if isinstance(first_item, list) and len(first_item) == 2:
                        # Format: [(text, {"entities": [...]}), ...]
                        training_data = json_data
                        print("✅ Données au format SpaCy direct détectées")
                    
                    # Si c'est au format JSON standard
                    elif isinstance(first_item, dict) and 'text' in first_item:
                        # Format: [{"text": "...", "entities": [...]}, ...]
                        for item in json_data:
                            if not isinstance(item, dict):
                                raise ValueError(f"Format d'item invalide: {type(item)}")
                            
                            text = item.get('text', '')
                            entities = item.get('entities', [])
                            
                            if not text:
                                print(f"⚠️ Texte vide ignoré: {item}")
                                continue
                                
                            training_data.append((text, {"entities": entities}))
                        print("✅ Données au format JSON standard converties")
                    
                    else:
                        raise ValueError(f"Format de données non reconnu. Premier item: {first_item}")
                
                else:
                    raise ValueError("Le fichier ne contient pas de données valides")
                
                if not training_data:
                    raise ValueError("Aucune donnée d'entraînement valide trouvée")
                
                # Validation des données
                valid_data = []
                for i, (text, annotations) in enumerate(training_data):
                    try:
                        # Vérifie que le texte est une chaîne
                        if not isinstance(text, str):
                            print(f"⚠️ Texte invalide à l'index {i}: {type(text)}")
                            continue
                        
                        # Vérifie que les annotations sont un dictionnaire
                        if not isinstance(annotations, dict):
                            print(f"⚠️ Annotations invalides à l'index {i}: {type(annotations)}")
                            continue
                        
                        # Vérifie que 'entities' existe dans les annotations
                        if 'entities' not in annotations:
                            annotations = {"entities": []}
                        
                        # Vérifie le format des entités
                        entities = annotations['entities']
                        if not isinstance(entities, list):
                            print(f"⚠️ Liste d'entités invalide à l'index {i}: {type(entities)}")
                            continue
                        
                        # Vérifie chaque entité
                        valid_entities = []
                        for entity in entities:
                            if isinstance(entity, (list, tuple)) and len(entity) == 3:
                                start, end, label = entity
                                if (isinstance(start, int) and isinstance(end, int) and 
                                    isinstance(label, str) and 0 <= start < end <= len(text)):
                                    valid_entities.append((start, end, label))
                                else:
                                    print(f"⚠️ Entité invalide ignorée: {entity}")
                            else:
                                print(f"⚠️ Format d'entité invalide: {entity}")
                        
                        valid_data.append((text, {"entities": valid_entities}))
                        
                    except Exception as e:
                        print(f"⚠️ Erreur lors de la validation de l'item {i}: {e}")
                        continue
                
                if not valid_data:
                    raise ValueError("Aucune donnée valide après validation")
                
                self.generated_training_data = valid_data
                self.training_data_path = filepath
                
                self.data_status_label.config(
                    text=f"✅ {len(valid_data)} exemples chargés", 
                    fg="green"
                )
                
                # Affiche un aperçu des données chargées
                if len(valid_data) != len(training_data):
                    messagebox.showwarning("Données filtrées", 
                                         f"{len(training_data)} exemples dans le fichier\n"
                                         f"{len(valid_data)} exemples valides chargés\n"
                                         f"{len(training_data) - len(valid_data)} exemples ignorés")
                
                self.update_status(f"Données d'entraînement chargées: {len(valid_data)} exemples")
                
                # Optionnel: Affiche un aperçu
                if messagebox.askyesno("Aperçu", "Voulez-vous voir un aperçu des données chargées ?"):
                    preview = self.data_generator.preview_training_data(valid_data, max_examples=3)
                    messagebox.showinfo("Aperçu des données", preview)
                
            except json.JSONDecodeError as e:
                messagebox.showerror("Erreur de format", 
                                   f"Le fichier n'est pas un JSON valide:\n{e}")
            except Exception as e:
                messagebox.showerror("Erreur de chargement", 
                                   f"Impossible de charger le fichier:\n{e}")
    
    # ======================
    # MÉTHODES D'ENTRAÎNEMENT
    # ======================
    
    def start_training(self):
        """
        Lance l'entraînement du modèle avec les données chargées
        """
        if not self.generated_training_data:
            messagebox.showwarning("Données manquantes", 
                                 "Veuillez d'abord générer ou charger des données d'entraînement")
            return
        if not isinstance(self.generated_training_data, list):
            messagebox.showerror("Erreur de format", 
                           "Les données d'entraînement ne sont pas au bon format")
        return
    
        if len(self.generated_training_data) == 0:
            messagebox.showerror("Données vides", 
                           "Aucune donnée d'entraînement disponible")
        return  

      # Vérifie le format du premier élément
        try:
            first_item = self.generated_training_data[0]
            if not (isinstance(first_item, (list, tuple)) and len(first_item) == 2):
                raise ValueError("Format d'item invalide")
        
            text, annotations = first_item
            if not isinstance(text, str) or not isinstance(annotations, dict):
                raise ValueError("Types d'item invalides")
            
        except (IndexError, ValueError) as e:
            messagebox.showerror("Erreur de format", 
                           f"Format des données d'entraînement invalide: {e}")
            return
        
        if not self.custom_entities:
            messagebox.showwarning("Configuration manquante", 
                             "Veuillez d'abord configurer vos entités personnalisées")
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
            training_thread = threading.Thread(
                target=self._run_training,
                args=(training_config,)
            )
            training_thread.daemon = True
            training_thread.start()
            
        except Exception as e:
            messagebox.showerror("Erreur d'entraînement", f"Erreur lors du lancement: {e}")

    def _run_training(self, config):
        """
        Exécute l'entraînement dans un thread séparé
        
        Args:
            config: Configuration d'entraînement
        """
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

    def _handle_training_results(self, results):
        """
        Traite les résultats de l'entraînement
        
        Args:
            results: Résultats retournés par le trainer
        """
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
        """
        Ajoute un message au log d'entraînement
        
        Args:
            message: Message à ajouter
        """
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
    
    # ======================
    # MÉTHODES DE PSEUDONYMISATION
    # ======================
    
    def pseudonymize_text(self):
        """
        Pseudonymise le texte saisi par l'utilisateur
        """
        if not self.trained_model_path:
            messagebox.showwarning("Modèle manquant", 
                                 "Veuillez d'abord sélectionner un modèle entraîné")
            return
        
        # Récupère le texte à pseudonymiser
        input_text = self.input_text.get(1.0, tk.END).strip()
        
        if not input_text:
            messagebox.showwarning("Texte manquant", 
                                 "Veuillez saisir un texte à pseudonymiser")
            return
        
        try:
            # Initialise le pseudonymiseur si nécessaire
            if not self.pseudonymizer:
                self.pseudonymizer = TextPseudonymizer()
                if not self.pseudonymizer.load_model(self.trained_model_path):
                    messagebox.showerror("Erreur de modèle", 
                                       "Impossible de charger le modèle pour la pseudonymisation")
                    return
            
            # Dialogue pour sélectionner les types d'entités à masquer
            entity_selection = EntityMaskingDialog(self.root, self.custom_entities)
            selected_entities = entity_selection.result
            
            if selected_entities is None:  # Annulé
                return
            
            # Prévisualisation optionnelle
            if messagebox.askyesno("Prévisualisation", 
                                 "Voulez-vous prévisualiser les entités qui seront pseudonymisées ?"):
                try:
                    preview = self.pseudonymizer.preview_pseudonymization(
                        input_text, selected_entities if selected_entities else None
                    )
                    
                    preview_text = self._format_preview_text(preview)
                    
                    # Affiche la prévisualisation
                    if not messagebox.askyesno("Continuer la pseudonymisation", preview_text):
                        return
                        
                except Exception as e:
                    messagebox.showerror("Erreur de prévisualisation", f"Erreur: {e}")
                    return
            
            # Effectue la pseudonymisation
            self.update_status("Pseudonymisation en cours...")
            
            pseudonymized_text, stats = self.pseudonymizer.pseudonymize_text(
                input_text,
                entity_types_to_mask=selected_entities if selected_entities else None,
                preserve_format=True
            )
            
            # Affiche le résultat
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(1.0, pseudonymized_text)
            
            # Affiche les statistiques
            stats_message = self._format_pseudonymization_stats(stats)
            
            # Propose de sauvegarder le fichier de correspondance
            if messagebox.askyesno("Sauvegarde des correspondances", 
                                 f"Pseudonymisation terminée!\n\n{stats_message}\n\n"
                                 "Voulez-vous sauvegarder le fichier de correspondance ?"):
                self.save_correspondence_file(stats)
            
            self.update_status(f"Pseudonymisation terminée - {stats['entities_processed']} entités traitées")
            
        except Exception as e:
            messagebox.showerror("Erreur de pseudonymisation", f"Erreur: {e}")
            self.update_status("Erreur lors de la pseudonymisation")

    def _format_preview_text(self, preview):
        """
        Formate le texte de prévisualisation de la pseudonymisation
        
        Args:
            preview: Données de prévisualisation
            
        Returns:
            str: Texte formaté pour l'affichage
        """
        text = f"PRÉVISUALISATION DE LA PSEUDONYMISATION\n{'='*50}\n\n"
        text += f"Total d'entités détectées: {preview['total_entities']}\n"
        text += f"Nouveaux pseudonymes à créer: {preview['would_create_pseudonyms']}\n"
        text += f"Pseudonymes existants réutilisés: {preview['would_reuse_pseudonyms']}\n\n"
        
        if preview['entities_by_type']:
            text += "RÉPARTITION PAR TYPE D'ENTITÉ:\n"
            text += "-" * 30 + "\n"
            for entity_type, count in preview['entities_by_type'].items():
                text += f"{entity_type}: {count} entité(s)\n"
            text += "\n"
        
        if preview['entities_details']:
            text += "DÉTAIL DES ENTITÉS (premières 10):\n"
            text += "-" * 35 + "\n"
            for i, entity in enumerate(preview['entities_details'][:10]):
                status = "NOUVEAU" if entity['is_new'] else "EXISTANT"
                text += f"{i+1}. '{entity['original']}' ({entity['type']}) → {entity['pseudonym']} [{status}]\n"
            
            if len(preview['entities_details']) > 10:
                text += f"... et {len(preview['entities_details']) - 10} autres entités\n"
        
        text += "\nVoulez-vous continuer avec la pseudonymisation ?"
        return text

    def _format_pseudonymization_stats(self, stats):
        """
        Formate les statistiques de pseudonymisation
        
        Args:
            stats: Statistiques de pseudonymisation
            
        Returns:
            str: Texte formaté
        """
        text = f"Entités traitées: {stats['entities_processed']}\n"
        text += f"Nouveaux pseudonymes créés: {stats['pseudonyms_created']}\n"
        text += f"Pseudonymes réutilisés: {stats['pseudonyms_reused']}\n"
        
        if stats['entities_by_type']:
            text += "\nRépartition par type:\n"
            for entity_type, count in stats['entities_by_type'].items():
                text += f"- {entity_type}: {count}\n"
        
        return text

    def save_correspondence_file(self, pseudonymization_stats):
        """
        Sauvegarde le fichier de correspondance
        
        Args:
            pseudonymization_stats: Statistiques de la pseudonymisation
        """
        try:
            # Dialogue pour choisir l'emplacement
            filepath = filedialog.asksaveasfilename(
                title="Sauvegarder le fichier de correspondance",
                defaultextension=".json",
                filetypes=[("Fichiers JSON", "*.json"), ("Tous les fichiers", "*.*")]
            )
            
            if filepath:
                # Informations supplémentaires à inclure
                additional_info = {
                    'pseudonymization_stats': pseudonymization_stats,
                    'model_used': self.trained_model_path,
                    'custom_entities': self.custom_entities,
                    'creation_context': 'manual_pseudonymization'
                }
                
                # Sauvegarde
                saved_path = self.pseudonymizer.save_correspondence_file(filepath, additional_info)
                self.correspondence_file_path = saved_path
                
                messagebox.showinfo("Sauvegarde réussie", 
                                  f"Fichier de correspondance sauvegardé:\n{saved_path}")
                
        except Exception as e:
            messagebox.showerror("Erreur de sauvegarde", f"Erreur: {e}")

    def copy_to_depseudo(self):
        """
        Copie le texte pseudonymisé vers l'onglet de dépseudonymisation
        """
        pseudonymized_text = self.output_text.get(1.0, tk.END).strip()
        if pseudonymized_text:
            self.pseudo_input_text.delete(1.0, tk.END)
            self.pseudo_input_text.insert(1.0, pseudonymized_text)
            
            # Passe à l'onglet de dépseudonymisation
            self.notebook.select(4)  # Index de l'onglet dépseudonymisation
            
            messagebox.showinfo("Copie effectuée", "Texte copié vers l'onglet de dépseudonymisation")
        else:
            messagebox.showwarning("Aucun contenu", "Aucun texte pseudonymisé à copier")
    
    # ======================
    # MÉTHODES DE DÉPSEUDONYMISATION
    # ======================
    
    def load_correspondence_file(self):
        """
        Charge un fichier de correspondance pour la dépseudonymisation
        """
        filepath = filedialog.askopenfilename(
            title="Sélectionner le fichier de correspondance",
            filetypes=[("Fichiers JSON", "*.json"), ("Tous les fichiers", "*.*")]
        )
        
        if filepath:
            try:
                # Initialise le pseudonymiseur si nécessaire
                if not self.pseudonymizer:
                    self.pseudonymizer = TextPseudonymizer()
                
                # Charge le fichier de correspondance
                if self.pseudonymizer.load_correspondence_file(filepath):
                    self.correspondence_file_path = filepath
                    
                    # Met à jour l'affichage
                    filename = Path(filepath).name
                    self.corresp_status_label.config(
                        text=f"✅ Fichier chargé: {filename}", 
                        fg="green"
                    )
                    
                    # Affiche les informations du fichier
                    summary = self.pseudonymizer.get_pseudonymization_summary()
                    info_text = (f"Correspondances chargées: {summary['total_pseudonyms']}\n"
                                f"Types d'entités: {', '.join(summary['entity_types_processed'])}")
                    
                    messagebox.showinfo("Fichier chargé", info_text)
                    
                    self.update_status(f"Correspondances chargées: {summary['total_pseudonyms']} pseudonymes")
                    
                else:
                    messagebox.showerror("Erreur", "Impossible de charger le fichier de correspondance")
                    
            except Exception as e:
                messagebox.showerror("Erreur de chargement", f"Erreur: {e}")

    def depseudonymize_text(self):
        """
        Dépseudonymise le texte saisi par l'utilisateur
        """
        if not self.correspondence_file_path and not (self.pseudonymizer and self.pseudonymizer.correspondence_map):
            messagebox.showwarning("Correspondances manquantes", 
                                 "Veuillez d'abord charger un fichier de correspondance")
            return
        
        # Récupère le texte pseudonymisé
        pseudo_text = self.pseudo_input_text.get(1.0, tk.END).strip()
        
        if not pseudo_text:
            messagebox.showwarning("Texte manquant", 
                                 "Veuillez saisir un texte pseudonymisé à restaurer")
            return
        
        try:
            # Initialise le pseudonymiseur si nécessaire
            if not self.pseudonymizer:
                self.pseudonymizer = TextPseudonymizer()
                
                # Charge le fichier de correspondance
                if self.correspondence_file_path:
                    if not self.pseudonymizer.load_correspondence_file(self.correspondence_file_path):
                        messagebox.showerror("Erreur", "Impossible de charger les correspondances")
                        return
            
            # Effectue la dépseudonymisation
            self.update_status("Dépseudonymisation en cours...")
            
            depseudonymized_text = self.pseudonymizer.depseudonymize_text(pseudo_text)
            
            # Affiche le résultat
            self.depseudo_output_text.delete(1.0, tk.END)
            self.depseudo_output_text.insert(1.0, depseudonymized_text)
            
            # Calcule quelques statistiques
            original_pseudonyms = len([word for word in pseudo_text.split() 
                                     if any(word.startswith(prefix) for prefix in 
                                           ['PERS_', 'ETAB_', 'ORG_', 'LIEU_', 'CODE_'])])
            
            messagebox.showinfo("Dépseudonymisation terminée", 
                              f"Texte restauré avec succès!\n"
                              f"Pseudonymes détectés et remplacés: {original_pseudonyms} (estimation)")
            
            self.update_status("Dépseudonymisation terminée avec succès")
            
        except Exception as e:
            messagebox.showerror("Erreur de dépseudonymisation", f"Erreur: {e}")
            self.update_status("Erreur lors de la dépseudonymisation")
    
    # ======================
    # MÉTHODES UTILITAIRES
    # ======================
    
    def import_text_file(self, text_widget):
        """
        Importe un fichier texte dans un widget de texte
        
        Args:
            text_widget: Widget de texte où insérer le contenu
        """
        filepath = filedialog.askopenfilename(
            title="Importer un fichier texte",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                text_widget.delete(1.0, tk.END)
                text_widget.insert(1.0, content)
                
                self.update_status(f"Fichier importé: {Path(filepath).name}")
                
            except Exception as e:
                messagebox.showerror("Erreur d'importation", f"Erreur: {e}")

    def export_text_file(self, text_widget, default_name="exported_text.txt"):
        """
        Exporte le contenu d'un widget de texte vers un fichier
        
        Args:
            text_widget: Widget de texte à exporter
            default_name: Nom de fichier par défaut
        """
        content = text_widget.get(1.0, tk.END).strip()
        
        if not content:
            messagebox.showwarning("Contenu vide", "Aucun contenu à exporter")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Exporter vers un fichier",
            defaultextension=".txt",
            initialvalue=default_name,
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                messagebox.showinfo("Export réussi", f"Contenu exporté vers:\n{filepath}")
                self.update_status(f"Fichier exporté: {Path(filepath).name}")
                
            except Exception as e:
                messagebox.showerror("Erreur d'export", f"Erreur: {e}")
    
    def update_status(self, message):
        """Met à jour la barre de statut"""
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
# Fichier : main.py (version corrigée)

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
from typing import Dict
from pdf_analyzer import PDFAnalyzer

# Ajout du dossier modules au chemin Python
# Assurez-vous que le script est lancé depuis le dossier 'pseudonymization_app'
# ou ajustez le chemin en conséquence.
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Import des modules personnalisés
try:
    from data_generator import TrainingDataGenerator
    from model_trainer import SpacyModelTrainer
    from pseudonymizer import TextPseudonymizer
    from utils import AppUtils
    # On importe l'analyseur de document de manière conditionnelle
    try:
        from document_analyzer import DocumentAnalyzer
    except ImportError:
        DocumentAnalyzer = None
        print("⚠️ Le module 'document_analyzer' ou ses dépendances (python-docx) ne sont pas trouvés.")
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
        
        # Initialisation conditionnelle de l'analyseur de document
        if DocumentAnalyzer:
            self.document_analyzer = DocumentAnalyzer()
            self.loaded_document = None
            self.document_analysis = None
        else:
            self.document_analyzer = None

        self.setup_ui()
        self.create_directories()

        try:
            self.pdf_analyzer = PDFAnalyzer()
            self.loaded_pdf = None
            self.pdf_analysis = None
        except ImportError as e:
            self.pdf_analyzer = None
            print(f"⚠️ Analyseur PDF non disponible: {e}")
        
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
        """ Crée l'onglet d'entraînement. """
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
    # """
    # Crée l'onglet de pseudonymisation des textes (VERSION MISE À JOUR AVEC PDF)
    # """
        pseudo_frame = ttk.Frame(self.notebook)
        self.notebook.add(pseudo_frame, text="4. Pseudonymisation")
        
        # Titre
        title_label = tk.Label(pseudo_frame, text="Pseudonymisation de Texte et Documents", 
                            font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Sous-onglets pour différents types de contenu
        pseudo_notebook = ttk.Notebook(pseudo_frame)
        pseudo_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Onglet pseudonymisation de texte simple
        text_frame = ttk.Frame(pseudo_notebook)
        pseudo_notebook.add(text_frame, text="Texte Simple")
        self.create_text_pseudonymization_interface(text_frame)
        
        # Onglet pseudonymisation de documents Word
        if self.document_analyzer:
            doc_frame = ttk.Frame(pseudo_notebook)
            pseudo_notebook.add(doc_frame, text="Documents Word (.docx)")
            self.create_document_pseudonymization_interface(doc_frame)
        
        # Onglet pseudonymisation de documents PDF
        if self.pdf_analyzer:
            pdf_frame = ttk.Frame(pseudo_notebook)
            pseudo_notebook.add(pdf_frame, text="Documents PDF (.pdf)")
            self.create_pdf_pseudonymization_interface(pdf_frame)

    def create_pdf_pseudonymization_interface(self, parent_frame):
    # """
    # Crée l'interface de pseudonymisation pour les documents PDF
    # """
    # Section chargement du document PDF
        load_frame = ttk.LabelFrame(parent_frame, text="Chargement du Document PDF")
        load_frame.pack(fill=tk.X, padx=20, pady=10)
        
        load_pdf_button = tk.Button(load_frame, text="Charger Document PDF (.pdf)",
                                command=self.load_pdf_document, bg="#4CAF50", fg="white")
        load_pdf_button.pack(pady=10)
        
        self.pdf_status_label = tk.Label(load_frame, text="Aucun document PDF chargé", fg="red")
        self.pdf_status_label.pack()
        
        # Boutons d'analyse et aperçu
        pdf_analysis_frame = tk.Frame(load_frame)
        pdf_analysis_frame.pack(pady=10)
        
        analyze_pdf_button = tk.Button(pdf_analysis_frame, text="Analyser le PDF",
                                    command=self.analyze_pdf_document, bg="#2196F3", fg="white")
        analyze_pdf_button.pack(side=tk.LEFT, padx=5)
        
        preview_pdf_button = tk.Button(pdf_analysis_frame, text="Aperçu du PDF",
                                    command=self.preview_pdf_document, bg="#607D8B", fg="white")
        preview_pdf_button.pack(side=tk.LEFT, padx=5)
        
        metadata_button = tk.Button(pdf_analysis_frame, text="Métadonnées",
                                command=self.show_pdf_metadata, bg="#795548", fg="white")
        metadata_button.pack(side=tk.LEFT, padx=5)
        
        # Zone de résultats d'analyse PDF
        pdf_analysis_frame = ttk.LabelFrame(parent_frame, text="Résultats d'Analyse PDF")
        pdf_analysis_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.pdf_analysis_text = scrolledtext.ScrolledText(pdf_analysis_frame, height=8, wrap=tk.WORD)
        self.pdf_analysis_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Boutons d'action sur le PDF
        pdf_actions_frame = tk.Frame(parent_frame)
        pdf_actions_frame.pack(pady=10)
        
        pseudo_pdf_button = tk.Button(pdf_actions_frame, text="Pseudonymiser PDF",
                                    command=self.pseudonymize_pdf_document,
                                    bg="#FF9800", fg="white", font=("Arial", 11))
        pseudo_pdf_button.pack(side=tk.LEFT, padx=5)
        
        annotate_pdf_button = tk.Button(pdf_actions_frame, text="Créer PDF Annoté",
                                    command=self.create_annotated_pdf,
                                    bg="#E91E63", fg="white", font=("Arial", 11))
        annotate_pdf_button.pack(side=tk.LEFT, padx=5)
        
        export_pdf_report_button = tk.Button(pdf_actions_frame, text="Exporter Rapport",
                                            command=self.export_pdf_analysis_report,
                                            bg="#9C27B0", fg="white", font=("Arial", 11))
        export_pdf_report_button.pack(side=tk.LEFT, padx=5)

    def load_pdf_document(self):
        """
        Charge un document PDF pour analyse et pseudonymisation
        """
        if not self.pdf_analyzer:
            messagebox.showerror("Erreur", "L'analyseur PDF n'est pas disponible.\n"
                            "Installez PyMuPDF avec: pip install PyMuPDF")
            return
        
        filepath = filedialog.askopenfilename(
            title="Sélectionner un document PDF",
            filetypes=[("Documents PDF", "*.pdf"), ("Tous les fichiers", "*.*")]
        )
        
        if filepath:
            try:
                # Charge le document PDF
                if self.pdf_analyzer.load_document(filepath):
                    self.loaded_pdf = filepath
                    self.pdf_status_label.config(
                        text=f"✅ PDF chargé: {Path(filepath).name}",
                        fg="green"
                    )
                    self.update_status(f"Document PDF chargé: {Path(filepath).name}")
                    
                    # Affiche les informations de base
                    if self.pdf_analyzer.document:
                        page_count = self.pdf_analyzer.document.page_count
                        self.pdf_status_label.config(
                            text=f"✅ PDF chargé: {Path(filepath).name} ({page_count} pages)",
                            fg="green"
                        )
                else:
                    messagebox.showerror("Erreur", "Impossible de charger le document PDF")
            except Exception as e:
                messagebox.showerror("Erreur de chargement", f"Erreur: {e}")

    def analyze_pdf_document(self):
        """
        Lance l'analyse du document PDF chargé
        """
        if not self.loaded_pdf or not self.pdf_analyzer:
            messagebox.showwarning("Document requis", "Veuillez d'abord charger un document PDF")
            return
        
        if not hasattr(self, 'pseudonymizer') or not self.pseudonymizer:
            messagebox.showwarning("Modèle requis", "Veuillez d'abord sélectionner un modèle entraîné")
            return
        
        try:
            # Affiche un dialogue de progression
            progress_dialog = ProgressDialog(self.root, "Analyse du PDF en cours...")
            
            # Extrait la structure du document
            document_info = self.pdf_analyzer.extract_text_with_structure()
            
            # Analyse les entités avec le modèle chargé
            entities_analysis = self.pdf_analyzer.analyze_entities_in_document(
                self.pseudonymizer.nlp,
                entity_types=self.custom_entities if self.custom_entities else None
            )
            
            progress_dialog.destroy()
            
            # Sauvegarde l'analyse
            self.pdf_analysis = {
                'document_info': document_info,
                'entities_analysis': entities_analysis
            }
            
            # Affiche les résultats
            self.display_pdf_analysis()
            
            messagebox.showinfo("Analyse PDF terminée", 
                            f"Document PDF analysé avec succès!\n"
                            f"Pages: {document_info['page_count']}\n"
                            f"Entités trouvées: {entities_analysis['total_entities']}\n"
                            f"Entités uniques: {len(entities_analysis['unique_entities'])}")
            
        except Exception as e:
            if 'progress_dialog' in locals():
                progress_dialog.destroy()
            messagebox.showerror("Erreur d'analyse PDF", f"Erreur lors de l'analyse: {e}")

    def display_pdf_analysis(self):
        """
        Affiche les résultats de l'analyse du document PDF
        """
        if not self.pdf_analysis:
            return
        
        doc_info = self.pdf_analysis['document_info']
        entities_info = self.pdf_analysis['entities_analysis']
        
        # Construit le texte d'affichage
        display_text = f"📄 ANALYSE DU DOCUMENT PDF: {doc_info['filename']}\n"
        display_text += "=" * 70 + "\n\n"
        
        # Statistiques générales
        display_text += "📊 STATISTIQUES GÉNÉRALES:\n"
        display_text += f"• Pages: {doc_info['page_count']}\n"
        display_text += f"• Mots total: {doc_info['total_words']}\n"
        display_text += f"• Tableaux détectés: {doc_info['tables_detected']}\n"
        display_text += f"• Images détectées: {doc_info['images_detected']}\n\n"
        
        # Métadonnées si disponibles
        if doc_info.get('metadata'):
            display_text += "📋 MÉTADONNÉES:\n"
            metadata = doc_info['metadata']
            for key, value in metadata.items():
                if value and key in ['title', 'author', 'subject', 'creator']:
                    display_text += f"• {key.title()}: {value}\n"
            display_text += "\n"
        
        # Entités identifiées
        display_text += "🎯 ENTITÉS IDENTIFIÉES:\n"
        display_text += f"• Total d'entités: {entities_info['total_entities']}\n"
        display_text += f"• Entités uniques: {len(entities_info['unique_entities'])}\n"
        display_text += f"• Pages avec entités: {len(entities_info.get('entities_by_page', {}))}\n\n"
        
        # Répartition par page
        entities_by_page = entities_info.get('entities_by_page', {})
        if entities_by_page:
            display_text += "📄 RÉPARTITION PAR PAGE:\n"
            for page_num in sorted(entities_by_page.keys()):
                entities = entities_by_page[page_num]
                entity_types = {}
                for entity in entities:
                    entity_type = entity['label']
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                
                display_text += f"• Page {page_num}: {len(entities)} entités ("
                display_text += ", ".join([f"{t}:{c}" for t, c in entity_types.items()])
                display_text += ")\n"
            display_text += "\n"
        
        # Détail par type d'entité
        entities_by_type = entities_info.get('entities_by_type', {})
        if entities_by_type:
            display_text += "📋 RÉPARTITION PAR TYPE:\n"
            for entity_type, entities in entities_by_type.items():
                unique_entities = list(set(entities))
                display_text += f"• {entity_type}: {len(entities)} occurrences, {len(unique_entities)} uniques\n"
                
                # Affiche quelques exemples
                for entity in unique_entities[:3]:
                    count = entities.count(entity)
                    display_text += f"  - {entity} ({count}x)\n"
                
                if len(unique_entities) > 3:
                    display_text += f"  ... et {len(unique_entities) - 3} autres\n"
                display_text += "\n"
        
        # Aperçu du contenu (première page)
        if doc_info.get('pages') and doc_info['pages']:
            first_page = doc_info['pages'][0]
            display_text += "👁️ APERÇU DE LA PREMIÈRE PAGE:\n"
            display_text += "-" * 50 + "\n"
            preview_text = first_page['text'][:800] if first_page['text'] else "Aucun texte extrait"
            if len(first_page['text']) > 800:
                preview_text += "\n[...texte tronqué...]"
            display_text += preview_text
        
        # Affiche dans la zone de texte
        self.pdf_analysis_text.delete(1.0, tk.END)
        self.pdf_analysis_text.insert(1.0, display_text)

    def preview_pdf_document(self):
        """
        Affiche un aperçu complet du document PDF
        """
        if not self.loaded_pdf or not self.pdf_analyzer:
            messagebox.showwarning("Document requis", "Veuillez d'abord charger un document PDF")
            return
        
        try:
            if not hasattr(self.pdf_analyzer, 'pages_content') or not self.pdf_analyzer.pages_content:
                # Extrait le texte si ce n'est pas déjà fait
                self.pdf_analyzer.extract_text_with_structure()
            
            preview_text = self.pdf_analyzer.get_document_preview(max_chars=2000, max_pages=5)
            
            # Affiche dans une nouvelle fenêtre
            preview_window = tk.Toplevel(self.root)
            preview_window.title(f"Aperçu PDF - {Path(self.loaded_pdf).name}")
            preview_window.geometry("900x700")
            
            preview_text_widget = scrolledtext.ScrolledText(preview_window, wrap=tk.WORD, font=("Courier", 10))
            preview_text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            preview_text_widget.insert(1.0, preview_text)
            preview_text_widget.config(state=tk.DISABLED)  # Lecture seule
            
            # Bouton pour fermer
            close_button = tk.Button(preview_window, text="Fermer", command=preview_window.destroy,
                                    bg="#f44336", fg="white")
            close_button.pack(pady=5)
            
        except Exception as e:
            messagebox.showerror("Erreur d'aperçu PDF", f"Erreur: {e}")

    def show_pdf_metadata(self):
        """
        Affiche les métadonnées du document PDF
        """
        if not self.loaded_pdf or not self.pdf_analyzer:
            messagebox.showwarning("Document requis", "Veuillez d'abord charger un document PDF")
            return
        
        try:
            metadata = self.pdf_analyzer.extract_metadata()
            
            # Crée une fenêtre pour afficher les métadonnées
            metadata_window = tk.Toplevel(self.root)
            metadata_window.title(f"Métadonnées PDF - {Path(self.loaded_pdf).name}")
            metadata_window.geometry("600x500")
            
            # Zone de texte pour les métadonnées
            metadata_text = scrolledtext.ScrolledText(metadata_window, wrap=tk.WORD, font=("Courier", 10))
            metadata_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Formate les métadonnées
            metadata_display = f"MÉTADONNÉES DU DOCUMENT PDF\n"
            metadata_display += "=" * 50 + "\n\n"
            
            # Informations de base
            basic_info = metadata.get('basic_info', {})
            if basic_info:
                metadata_display += "INFORMATIONS DE BASE:\n"
                metadata_display += "-" * 30 + "\n"
                for key, value in basic_info.items():
                    if value:
                        metadata_display += f"{key}: {value}\n"
                metadata_display += "\n"
            
            # Informations techniques
            metadata_display += "INFORMATIONS TECHNIQUES:\n"
            metadata_display += "-" * 30 + "\n"
            metadata_display += f"Nombre de pages: {metadata.get('page_count', 'N/A')}\n"
            metadata_display += f"Taille du fichier: {metadata.get('file_size', 0)} bytes\n"
            metadata_display += f"Document chiffré: {'Oui' if metadata.get('is_encrypted', False) else 'Non'}\n"
            metadata_display += f"Mot de passe requis: {'Oui' if metadata.get('needs_pass', False) else 'Non'}\n"
            metadata_display += "\n"
            
            # Permissions
            permissions = metadata.get('permissions', {})
            if permissions:
                metadata_display += "PERMISSIONS:\n"
                metadata_display += "-" * 30 + "\n"
                metadata_display += f"Impression: {'Autorisée' if permissions.get('can_print', True) else 'Interdite'}\n"
                metadata_display += f"Modification: {'Autorisée' if permissions.get('can_modify', True) else 'Interdite'}\n"
                metadata_display += f"Copie: {'Autorisée' if permissions.get('can_copy', True) else 'Interdite'}\n"
                metadata_display += f"Annotation: {'Autorisée' if permissions.get('can_annotate', True) else 'Interdite'}\n"
            
            metadata_text.insert(1.0, metadata_display)
            metadata_text.config(state=tk.DISABLED)
            
            # Bouton pour fermer
            close_button = tk.Button(metadata_window, text="Fermer", command=metadata_window.destroy,
                                    bg="#f44336", fg="white")
            close_button.pack(pady=5)
            
        except Exception as e:
            messagebox.showerror("Erreur métadonnées", f"Erreur: {e}")

    def pseudonymize_pdf_document(self):
        """
        Lance la pseudonymisation du document PDF
        """
        if not self.pdf_analysis or not self.pdf_analyzer:
            messagebox.showwarning("Analyse requise", "Veuillez d'abord analyser le document PDF")
            return
        
        if not hasattr(self, 'pseudonymizer') or not self.pseudonymizer:
            messagebox.showwarning("Modèle requis", "Veuillez sélectionner un modèle de pseudonymisation")
            return
        
        try:
            # Crée la correspondance de pseudonymisation
            entities_analysis = self.pdf_analysis['entities_analysis']
            pseudonymization_map = {}
            
            # Génère des pseudonymes pour chaque entité unique
            for entity in entities_analysis['unique_entities']:
                pseudonym = self.pseudonymizer.generate_pseudonym(entity)
                pseudonymization_map[entity] = pseudonym
            
            # Pseudonymise le document
            progress_dialog = ProgressDialog(self.root, "Pseudonymisation du PDF...")
            
            pseudonymized_pdf_path = self.pdf_analyzer.create_pseudonymized_pdf(
                pseudonymization_map, 
                highlight_changes=True
            )
            
            # Sauvegarde le fichier de correspondance
            correspondence_path = self.save_pdf_correspondence_file(pseudonymization_map)
            
            progress_dialog.destroy()
            
            messagebox.showinfo("Pseudonymisation PDF terminée", 
                            f"PDF pseudonymisé sauvegardé:\n{pseudonymized_pdf_path}\n\n"
                            f"Fichier de correspondance:\n{correspondence_path}")
            
            self.update_status("PDF pseudonymisé avec succès")
            
        except Exception as e:
            if 'progress_dialog' in locals():
                progress_dialog.destroy()
            messagebox.showerror("Erreur de pseudonymisation PDF", f"Erreur: {e}")

    def create_annotated_pdf(self):
        """
        Crée un PDF avec les entités surlignées
        """
        if not self.pdf_analysis or not self.pdf_analyzer:
            messagebox.showwarning("Analyse requise", "Veuillez d'abord analyser le document PDF")
            return
        
        try:
            progress_dialog = ProgressDialog(self.root, "Création du PDF annoté...")
            
            entities_analysis = self.pdf_analysis['entities_analysis']
            annotated_pdf_path = self.pdf_analyzer.create_annotated_pdf(entities_analysis)
            
            progress_dialog.destroy()
            
            messagebox.showinfo("PDF annoté créé", 
                            f"PDF avec annotations sauvegardé:\n{annotated_pdf_path}\n\n"
                            "Les entités sont surlignées par couleur selon leur type.")
            
            # Propose d'ouvrir le fichier
            if messagebox.askyesno("Ouvrir le PDF", "Voulez-vous ouvrir le PDF annoté ?"):
                os.startfile(annotated_pdf_path)  # Windows
            
        except Exception as e:
            if 'progress_dialog' in locals():
                progress_dialog.destroy()
            messagebox.showerror("Erreur création PDF annoté", f"Erreur: {e}")

    def export_pdf_analysis_report(self):
        """
        Exporte un rapport d'analyse détaillé du PDF
        """
        if not self.pdf_analysis or not self.pdf_analyzer:
            messagebox.showwarning("Analyse requise", "Veuillez d'abord analyser un document PDF")
            return
        
        try:
            report_path = self.pdf_analyzer.export_analysis_report()
            messagebox.showinfo("Rapport PDF exporté", f"Rapport d'analyse sauvegardé:\n{report_path}")
            
            # Propose d'ouvrir le rapport
            if messagebox.askyesno("Ouvrir le rapport", "Voulez-vous ouvrir le rapport maintenant ?"):
                os.startfile(report_path)  # Windows
                
        except Exception as e:
            messagebox.showerror("Erreur d'export PDF", f"Erreur lors de l'export: {e}")

    def save_pdf_correspondence_file(self, pseudonymization_map: Dict[str, str]) -> str:
        """
        Sauvegarde le fichier de correspondance pour la dépseudonymisation des PDF
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        correspondence_path = f"data/pdf_correspondence_{timestamp}.json"
        
        correspondence_data = {
            'timestamp': timestamp,
            'document_type': 'PDF',
            'document': Path(self.loaded_pdf).name if self.loaded_pdf else "unknown",
            'pages': self.pdf_analyzer.document.page_count if self.pdf_analyzer.document else 0,
            'correspondences': pseudonymization_map
        }
        
        with open(correspondence_path, 'w', encoding='utf-8') as f:
            json.dump(correspondence_data, f, indent=2, ensure_ascii=False)
        
        return correspondence_path

        def create_text_pseudonymization_interface(self, parent_frame):
            """ Crée l'interface pour la pseudonymisation de texte simple. """
            model_frame = ttk.LabelFrame(parent_frame, text="Sélection du Modèle")
            model_frame.pack(fill=tk.X, padx=20, pady=10)
            
            select_model_button = tk.Button(model_frame, text="Sélectionner Modèle Entraîné", command=self.select_trained_model, bg="#4CAF50", fg="white")
            select_model_button.pack(pady=10)
            
            self.model_status_label = tk.Label(model_frame, text="Aucun modèle sélectionné", fg="red")
            self.model_status_label.pack()
            
            input_frame = ttk.LabelFrame(parent_frame, text="Texte à Pseudonymiser")
            input_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            self.input_text = scrolledtext.ScrolledText(input_frame, height=8, wrap=tk.WORD)
            self.input_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            pseudo_button = tk.Button(parent_frame, text="Pseudonymiser Texte", command=self.pseudonymize_text, bg="#FF9800", fg="white", font=("Arial", 12))
            pseudo_button.pack(pady=10)
            
            output_frame = ttk.LabelFrame(parent_frame, text="Texte Pseudonymisé")
            output_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            self.output_text = scrolledtext.ScrolledText(output_frame, height=8, wrap=tk.WORD)
            self.output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        def create_document_pseudonymization_interface(self, parent_frame):
            """ Crée l'interface pour la pseudonymisation de documents Word. """
            load_frame = ttk.LabelFrame(parent_frame, text="Chargement du Document")
            load_frame.pack(fill=tk.X, padx=20, pady=10)
            
            load_doc_button = tk.Button(load_frame, text="Charger Document Word (.docx)", command=self.load_word_document, bg="#4CAF50", fg="white")
            load_doc_button.pack(pady=10)
            
            self.doc_status_label = tk.Label(load_frame, text="Aucun document chargé", fg="red")
            self.doc_status_label.pack()
            
            analysis_buttons_frame = tk.Frame(load_frame)
            analysis_buttons_frame.pack(pady=10)
            
            analyze_button = tk.Button(analysis_buttons_frame, text="Analyser le Document", command=self.analyze_document, bg="#2196F3", fg="white")
            analyze_button.pack(side=tk.LEFT, padx=5)
            
            preview_button = tk.Button(analysis_buttons_frame, text="Aperçu du Document", command=self.preview_document, bg="#607D8B", fg="white")
            preview_button.pack(side=tk.LEFT, padx=5)
            
            analysis_results_frame = ttk.LabelFrame(parent_frame, text="Résultats d'Analyse")
            analysis_results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            self.document_analysis_text = scrolledtext.ScrolledText(analysis_results_frame, height=8, wrap=tk.WORD)
            self.document_analysis_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            doc_actions_frame = tk.Frame(parent_frame)
            doc_actions_frame.pack(pady=10)
            
            pseudo_doc_button = tk.Button(doc_actions_frame, text="Pseudonymiser Document", command=self.pseudonymize_document, bg="#FF9800", fg="white", font=("Arial", 11))
            pseudo_doc_button.pack(side=tk.LEFT, padx=5)
            
            export_report_button = tk.Button(doc_actions_frame, text="Exporter Rapport d'Analyse", command=self.export_analysis_report, bg="#9C27B0", fg="white", font=("Arial", 11))
            export_report_button.pack(side=tk.LEFT, padx=5)

        def create_depseudonymization_tab(self):
            """ Crée l'onglet de dépseudonymisation (version corrigée). """
            depseudo_frame = ttk.Frame(self.notebook)
            self.notebook.add(depseudo_frame, text="5. Dépseudonymisation")

            title_label = tk.Label(depseudo_frame, text="Dépseudonymisation de Texte", font=("Arial", 16, "bold"))
            title_label.pack(pady=10)

            corresp_frame = ttk.LabelFrame(depseudo_frame, text="Fichier de Correspondance")
            corresp_frame.pack(fill=tk.X, padx=20, pady=10)

            load_corresp_button = tk.Button(corresp_frame, text="Charger Fichier de Correspondance", command=self.load_correspondence_file, bg="#4CAF50", fg="white")
            load_corresp_button.pack(pady=10)
            
            self.corresp_status_label = tk.Label(corresp_frame, text="Aucun fichier de correspondance chargé", fg="red")
            self.corresp_status_label.pack()

            input_frame = ttk.LabelFrame(depseudo_frame, text="Texte à Dépseudonymiser")
            input_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            self.pseudo_input_text = scrolledtext.ScrolledText(input_frame, height=8, wrap=tk.WORD)
            self.pseudo_input_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            depseudo_button = tk.Button(depseudo_frame, text="Dépseudonymiser", command=self.depseudonymize_text, bg="#2196F3", fg="white", font=("Arial", 12))
            depseudo_button.pack(pady=10)

            output_frame = ttk.LabelFrame(depseudo_frame, text="Texte Original Restauré")
            output_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            self.depseudo_output_text = scrolledtext.ScrolledText(output_frame, height=8, wrap=tk.WORD)
            self.depseudo_output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ==============================================================================
        # DÉBUT DES MÉTHODES DE LA CLASSE (INDENTATION CORRIGÉE)
        # ==============================================================================
        
        def load_word_document(self):
            """ Charge un document Word pour analyse et pseudonymisation. """
            if not self.document_analyzer:
                messagebox.showerror("Erreur", "L'analyseur de documents n'est pas disponible.\n"
                                "Installez python-docx avec: pip install python-docx")
                return
            
            filepath = filedialog.askopenfilename(
                title="Sélectionner un document Word",
                filetypes=[("Documents Word", "*.docx"), ("Tous les fichiers", "*.*")]
            )
            
            if filepath:
                try:
                    if self.document_analyzer.load_document(filepath):
                        self.loaded_document = filepath
                        self.doc_status_label.config(text=f"✅ Document chargé: {Path(filepath).name}", fg="green")
                        self.update_status(f"Document Word chargé: {Path(filepath).name}")
                    else:
                        messagebox.showerror("Erreur", "Impossible de charger le document")
                except Exception as e:
                    messagebox.showerror("Erreur de chargement", f"Erreur: {e}")

        def analyze_document(self):
            """ Lance l'analyse du document chargé. """
            if not self.loaded_document or not self.document_analyzer:
                messagebox.showwarning("Document requis", "Veuillez d'abord charger un document Word")
                return
            
            if not self.trained_model_path:
                messagebox.showwarning("Modèle requis", "Veuillez d'abord sélectionner un modèle entraîné pour l'analyse.")
                return

            if self.pseudonymizer is None:
                self.pseudonymizer = TextPseudonymizer(self.trained_model_path)
            
            try:
                progress_dialog = ProgressDialog(self.root, "Analyse du document en cours...")
                document_info = self.document_analyzer.extract_text_with_structure()
                entities_analysis = self.document_analyzer.analyze_entities_in_document(
                    self.pseudonymizer.nlp,
                    entity_types=self.custom_entities if self.custom_entities else None
                )
                progress_dialog.destroy()
                
                self.document_analysis = {'document_info': document_info, 'entities_analysis': entities_analysis}
                self.display_document_analysis()
                
                messagebox.showinfo("Analyse terminée", 
                                f"Document analysé avec succès!\n"
                                f"Entités trouvées: {entities_analysis['total_entities']}\n"
                                f"Entités uniques: {len(entities_analysis['unique_entities'])}")
            except Exception as e:
                if 'progress_dialog' in locals():
                    progress_dialog.destroy()
                messagebox.showerror("Erreur d'analyse", f"Erreur lors de l'analyse: {e}")

        def display_document_analysis(self):
            """ Affiche les résultats de l'analyse du document. """
            if not self.document_analysis:
                return
            
            doc_info = self.document_analysis['document_info']
            entities_info = self.document_analysis['entities_analysis']
            
            display_text = f"📄 ANALYSE DU DOCUMENT: {doc_info['filename']}\n"
            display_text += "=" * 60 + "\n\n"
            display_text += "📊 STATISTIQUES GÉNÉRALES:\n"
            display_text += f"• Paragraphes: {doc_info['total_paragraphs']}\n"
            display_text += f"• Tableaux: {doc_info['total_tables']}\n"
            display_text += f"• Mots total: {doc_info['total_words']}\n\n"
            display_text += "🎯 ENTITÉS IDENTIFIÉES:\n"
            display_text += f"• Total d'entités: {entities_info['total_entities']}\n"
            display_text += f"• Entités uniques: {len(entities_info['unique_entities'])}\n\n"
            
            if entities_info['entities_by_type']:
                display_text += "📋 RÉPARTITION PAR TYPE:\n"
                for entity_type, entities in entities_info['entities_by_type'].items():
                    unique_entities = list(set(entities))
                    display_text += f"• {entity_type}: {len(entities)} occurrences, {len(unique_entities)} uniques\n"
                    for entity in unique_entities[:5]:
                        count = entities.count(entity)
                        display_text += f"  - {entity} ({count}x)\n"
                    if len(unique_entities) > 5:
                        display_text += f"  ... et {len(unique_entities) - 5} autres\n"
                    display_text += "\n"
            
            self.document_analysis_text.delete(1.0, tk.END)
            self.document_analysis_text.insert(1.0, display_text)

        def preview_document(self):
            """ Affiche un aperçu du document chargé. """
            if not self.loaded_document or not self.document_analyzer:
                messagebox.showwarning("Document requis", "Veuillez d'abord charger un document Word")
                return
            
            try:
                if not hasattr(self.document_analyzer, 'extracted_text') or not self.document_analyzer.extracted_text:
                    self.document_analyzer.extract_text_with_structure()
                
                preview_text = self.document_analyzer.get_document_preview(2000)
                
                preview_window = tk.Toplevel(self.root)
                preview_window.title(f"Aperçu - {Path(self.loaded_document).name}")
                preview_window.geometry("800x600")
                
                preview_text_widget = scrolledtext.ScrolledText(preview_window, wrap=tk.WORD)
                preview_text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                preview_text_widget.insert(1.0, preview_text)
                preview_text_widget.config(state=tk.DISABLED)
                
            except Exception as e:
                messagebox.showerror("Erreur d'aperçu", f"Erreur: {e}")

        def pseudonymize_document(self):
            """ Lance la pseudonymisation du document Word. """
            if not self.document_analysis or not self.document_analyzer:
                messagebox.showwarning("Analyse requise", "Veuillez d'abord analyser le document")
                return
            
            if self.pseudonymizer is None:
                messagebox.showwarning("Modèle requis", "Veuillez sélectionner un modèle de pseudonymisation")
                return
            
            try:
                entities_analysis = self.document_analysis['entities_analysis']
                pseudonymization_map = {}
                
                for entity in entities_analysis['unique_entities']:
                    pseudonym = self.pseudonymizer.generate_pseudonym(entity, 'UNKNOWN') # Le type n'est pas connu ici, fallback
                    pseudonymization_map[entity] = pseudonym
                
                progress_dialog = ProgressDialog(self.root, "Pseudonymisation du document...")
                
                pseudonymized_doc = self.document_analyzer.pseudonymize_document(pseudonymization_map, highlight_changes=True)
                output_path = self.document_analyzer.save_pseudonymized_document(pseudonymized_doc)
                correspondence_path = self.save_correspondence_file(pseudonymization_map)
                
                progress_dialog.destroy()
                
                messagebox.showinfo("Pseudonymisation terminée", 
                                f"Document pseudonymisé sauvegardé:\n{output_path}\n\n"
                                f"Fichier de correspondance:\n{correspondence_path}")
                self.update_status("Document pseudonymisé avec succès")
            except Exception as e:
                if 'progress_dialog' in locals():
                    progress_dialog.destroy()
                messagebox.showerror("Erreur de pseudonymisation", f"Erreur: {e}")

        def export_analysis_report(self):
            """ Exporte un rapport d'analyse détaillé. """
            if not self.document_analysis or not self.document_analyzer:
                messagebox.showwarning("Analyse requise", "Veuillez d'abord analyser un document")
                return
            
            try:
                report_path = self.document_analyzer.export_analysis_report()
                messagebox.showinfo("Rapport exporté", f"Rapport d'analyse sauvegardé:\n{report_path}")
                if messagebox.askyesno("Ouvrir le rapport", "Voulez-vous ouvrir le rapport maintenant ?"):
                    os.startfile(report_path)
            except Exception as e:
                messagebox.showerror("Erreur d'export", f"Erreur lors de l'export: {e}")

        def _save_correspondence_map(self, pseudonymization_map: Dict[str, str]) -> str:
            """ Sauvegarde le fichier de correspondance. """
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            correspondence_path = f"data/correspondence_{timestamp}.json"
            
            correspondence_data = {
                'timestamp': timestamp,
                'document': Path(self.loaded_document).name if self.loaded_document else "unknown",
                'correspondences': pseudonymization_map
            }
            
            with open(correspondence_path, 'w', encoding='utf-8') as f:
                json.dump(correspondence_data, f, indent=2, ensure_ascii=False)
            return correspondence_path

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

        def set_training_state(self, is_training):
            """ Active ou désactive les contrôles de l'interface pendant l'entraînement. """
            state = 'disabled' if is_training else 'normal'
            self.train_button.config(state=state)
            self.epochs_spinbox.config(state=state)
            self.batch_spinbox.config(state=state)
            for i, tab in enumerate(self.notebook.tabs()):
                if i != 2:
                    self.notebook.tab(i, state=state)

        def start_training(self):
            """ Lance l'entraînement du modèle. """
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
                    'dropout': 0.2, 'patience': 5, 'validation_split': 0.2
                }
                
                self.training_log.config(state='normal')
                self.training_log.delete(1.0, tk.END)
                self.log_training_message("🚀 Initialisation de l'entraînement...\n")
                self.training_log.config(state='disabled')

                training_thread = threading.Thread(target=self._run_training, args=(training_config,), daemon=True)
                training_thread.start()
            except Exception as e:
                messagebox.showerror("Erreur de lancement", f"Impossible de démarrer l'entraînement : {e}")

        def _run_training(self, config):
            """ Exécute l'entraînement dans un thread. """
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
                self.root.after(0, lambda err=e: messagebox.showerror("Erreur d'entraînement", str(err)))
            finally:
                self.training_in_progress = False
                self.root.after(0, self.set_training_state, False)

        def _handle_training_results(self, results):
            """ Traite les résultats à la fin de l'entraînement. """
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
        
        def log_training_message(self, message):
            """ Ajoute un message au log d'entraînement. """
            self.training_log.config(state='normal')
            self.training_log.insert(tk.END, message)
            self.training_log.see(tk.END)
            self.training_log.config(state='disabled')
            self.root.update_idletasks()

        def select_trained_model(self):
            """ Sélectionne un dossier contenant un modèle entraîné. """
            model_path = filedialog.askdirectory(title="Sélectionner le dossier du modèle entraîné")
            if not model_path:
                return

            try:
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
            """ Teste le modèle actuellement chargé. """
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

                entities_in_model = self.pseudonymizer.nlp.get_pipe("ner").labels
                entity_selection = EntityMaskingDialog(self.root, list(entities_in_model))
                if entity_selection.result is None:
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
            """ Met en forme les statistiques de pseudonymisation. """
            text = f"Entités traitées: {stats['entities_processed']}\n"
            text += f"Nouveaux pseudonymes: {stats['pseudonyms_created']}\n"
            text += f"Pseudonymes réutilisés: {stats['pseudonyms_reused']}"
            return text

        def save_correspondence_file(self, pseudonymization_stats):
            """ Sauvegarde le fichier de correspondance. """
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
            """ Charge un fichier de correspondance. """
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
            """ Dépseudonymise le texte. """
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
            """ Importe un fichier texte dans une zone de texte. """
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
            """ Exporte le contenu d'une zone de texte. """
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
            """ Met à jour la barre de statut. """
            self.status_bar.config(text=message)

# ======================
# CLASSES DE DIALOGUES
# ======================

class EntitySelectionDialog:
    """ Dialogue pour sélectionner un type d'entité. """
    def __init__(self, parent, entities):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Sélection du type d'entité")
        self.dialog.geometry("300x200")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        tk.Label(self.dialog, text="Sélectionnez le type d'entité pour ce fichier:", font=("Arial", 10)).pack(pady=10)
        
        self.selected_entity = tk.StringVar(value=entities[0] if entities else "")
        for entity in entities:
            rb = tk.Radiobutton(self.dialog, text=entity, variable=self.selected_entity, value=entity)
            rb.pack(anchor=tk.W, padx=20)
        
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=20)
        tk.Button(button_frame, text="OK", command=self.ok_clicked, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Annuler", command=self.cancel_clicked, bg="#f44336", fg="white").pack(side=tk.LEFT, padx=5)
        
        self.dialog.wait_window()
    
    def ok_clicked(self):
        self.result = self.selected_entity.get()
        self.dialog.destroy()
    
    def cancel_clicked(self):
        self.result = None
        self.dialog.destroy()

class EntityMaskingDialog:
    """ Dialogue pour sélectionner les types d'entités à pseudonymiser. """
    def __init__(self, parent, available_entities):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Sélection des entités à pseudonymiser")
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        tk.Label(self.dialog, text="Sélectionnez les types d'entités à pseudonymiser:", font=("Arial", 10)).pack(pady=10)
        
        checkboxes_frame = tk.Frame(self.dialog)
        checkboxes_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.entity_vars = {}
        self.all_entities_var = tk.BooleanVar(value=True)
        all_cb = tk.Checkbutton(checkboxes_frame, text="Toutes les entités", variable=self.all_entities_var, command=self.toggle_all_entities, font=("Arial", 10, "bold"))
        all_cb.pack(anchor=tk.W, pady=5)
        
        tk.Frame(checkboxes_frame, height=2, bg="gray").pack(fill=tk.X, pady=5)
        
        for entity in available_entities:
            var = tk.BooleanVar(value=True) # Par défaut, tout est coché
            self.entity_vars[entity] = var
            cb = tk.Checkbutton(checkboxes_frame, text=entity, variable=var, command=self.update_all_checkbox)
            cb.pack(anchor=tk.W, pady=2)
        
        self.toggle_all_entities() # Appeler pour synchroniser l'état initial
        
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=20)
        tk.Button(button_frame, text="OK", command=self.ok_clicked, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Annuler", command=self.cancel_clicked, bg="#f44336", fg="white").pack(side=tk.LEFT, padx=5)
        
        self.dialog.wait_window()
    
    def toggle_all_entities(self):
        is_checked = self.all_entities_var.get()
        for var in self.entity_vars.values():
            var.set(is_checked)
    
    def update_all_checkbox(self):
        all_selected = all(var.get() for var in self.entity_vars.values())
        self.all_entities_var.set(all_selected)
    
    def ok_clicked(self):
        self.result = [entity for entity, var in self.entity_vars.items() if var.get()]
        if not self.result:
            # Si rien n'est coché, on peut considérer que l'utilisateur ne veut rien masquer
            # ou qu'il veut tout masquer, cela dépend de la logique attendue. 
            # Renvoyer une liste vide semble plus sûr.
            self.result = []
        self.dialog.destroy()
    
    def cancel_clicked(self):
        self.result = None
        self.dialog.destroy()

class TestModelDialog:
    """ Dialogue pour tester un modèle entraîné. """
    def __init__(self, parent):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Test du Modèle")
        self.dialog.geometry("500x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        tk.Label(self.dialog, text="Saisissez un texte pour tester le modèle:", font=("Arial", 10)).pack(pady=10)
        
        text_frame = tk.Frame(self.dialog)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.text_widget = scrolledtext.ScrolledText(text_frame, height=8, wrap=tk.WORD)
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        
        example_text = ("Monsieur Dupont travaille à l'établissement ABC123. "
                       "L'organisation XYZ Corp a son siège à Paris. "
                       "Le code d'identification EST-456 correspond à notre filiale.")
        self.text_widget.insert(1.0, example_text)
        
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=20)
        tk.Button(button_frame, text="Tester", command=self.test_clicked, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Annuler", command=self.cancel_clicked, bg="#f44336", fg="white").pack(side=tk.LEFT, padx=5)
        
        self.dialog.wait_window()
    
    def test_clicked(self):
        self.result = self.text_widget.get(1.0, tk.END).strip()
        self.dialog.destroy()
    
    def cancel_clicked(self):
        self.result = None
        self.dialog.destroy()

class ProgressDialog:
    """ Dialogue de progression simple. """
    def __init__(self, parent, message):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Progression")
        self.dialog.geometry("400x100")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 100, parent.winfo_rooty() + 100))
        
        tk.Label(self.dialog, text=message, font=("Arial", 10)).pack(pady=10)
        
        self.progress = ttk.Progressbar(self.dialog, mode='indeterminate')
        self.progress.pack(pady=10, padx=20, fill=tk.X)
        self.progress.start()
        
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
    root = tk.Tk()
    app = PseudonymizationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
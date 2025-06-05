#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'entraînement de modèles SpaCy pour NER personnalisé
===========================================================

Ce module gère le fine-tuning de modèles SpaCy existants avec de nouvelles
entités NER personnalisées. Il inclut la gestion des callbacks de progression,
l'évaluation des performances et la sauvegarde des modèles entraînés.
"""

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Optional, Any
import numpy as np
from datetime import datetime
import pickle

class SpacyModelTrainer:
    """
    Gestionnaire d'entraînement pour les modèles SpaCy NER personnalisés
    
    Cette classe permet de :
    - Charger un modèle SpaCy de base
    - Ajouter de nouvelles entités NER
    - Effectuer le fine-tuning avec suivi des performances
    - Sauvegarder les modèles entraînés
    """
    
    def __init__(self, base_model_name: str = "fr_core_news_sm"):
        """
        Initialise le trainer avec un modèle de base
        
        Args:
            base_model_name (str): Nom du modèle SpaCy de base à utiliser
        """
        self.base_model_name = base_model_name
        self.nlp = None
        self.ner = None
        self.training_history = []
        self.custom_entities = []
        
        # Paramètres d'entraînement par défaut
        self.default_config = {
            'n_iter': 30,
            'dropout': 0.2,
            'batch_size': 8,
            'learn_rate': 0.001,
            'patience': 5,  # Pour l'early stopping
            'validation_split': 0.2
        }
        
        print(f"🤖 Trainer initialisé avec le modèle de base: {base_model_name}")
    
    def load_base_model(self) -> bool:
        """
        Charge le modèle SpaCy de base
        
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        try:
            print(f"📥 Chargement du modèle de base: {self.base_model_name}")
            
            # Tente de charger le modèle
            self.nlp = spacy.load(self.base_model_name)
            
            # Obtient le composant NER
            if "ner" in self.nlp.pipe_names:
                self.ner = self.nlp.get_pipe("ner")
                print("✅ Composant NER trouvé dans le modèle")
            else:
                # Crée un nouveau composant NER si absent
                self.ner = self.nlp.add_pipe("ner", last=True)
                print("🆕 Nouveau composant NER créé")
            
            return True
            
        except OSError as e:
            print(f"❌ Erreur: Le modèle '{self.base_model_name}' n'est pas installé")
            print(f"💡 Installez-le avec: python -m spacy download {self.base_model_name}")
            return False
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            return False
    
    def add_custom_entities(self, entity_types: List[str]) -> bool:
        """
        Ajoute de nouvelles étiquettes d'entités au modèle
        
        Args:
            entity_types (List[str]): Liste des nouveaux types d'entités
            
        Returns:
            bool: True si l'ajout a réussi, False sinon
        """
        if not self.nlp or not self.ner:
            print("❌ Modèle de base non chargé")
            return False
        
        try:
            print(f"🏷️ Ajout des entités personnalisées: {entity_types}")
            
            for entity_type in entity_types:
                # Ajoute l'étiquette au composant NER
                self.ner.add_label(entity_type)
                self.custom_entities.append(entity_type)
                print(f"  ✅ Entité ajoutée: {entity_type}")
            
            print(f"🎯 {len(entity_types)} nouvelles entités configurées")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de l'ajout des entités: {e}")
            return False
    
    def prepare_training_data(self, raw_training_data: List[Tuple[str, Dict]]) -> List[Example]:
        """
        Prépare les données d'entraînement au format SpaCy Example
        
        Args:
            raw_training_data: Données brutes (texte, annotations)
            
        Returns:
            List[Example]: Données au format SpaCy Example
        """
        examples = []
        
        print(f"🔄 Préparation de {len(raw_training_data)} exemples d'entraînement...")
        
        for text, annotations in raw_training_data:
            try:
                # Crée un document SpaCy
                doc = self.nlp.make_doc(text)
                
                # Crée l'exemple d'entraînement
                example = Example.from_dict(doc, annotations)
                examples.append(example)
                
            except Exception as e:
                print(f"⚠️ Erreur avec l'exemple '{text[:50]}...': {e}")
                continue
        
        print(f"✅ {len(examples)} exemples préparés avec succès")
        return examples
    
    def split_data(self, examples: List[Example], validation_split: float = 0.2) -> Tuple[List[Example], List[Example]]:
        """
        Divise les données en ensembles d'entraînement et de validation
        
        Args:
            examples: Liste des exemples
            validation_split: Proportion pour la validation
            
        Returns:
            Tuple: (données d'entraînement, données de validation)
        """
        # Mélange les données
        random.shuffle(examples)
        
        # Calcule la taille de validation
        val_size = int(len(examples) * validation_split)
        
        # Divise les données
        train_examples = examples[val_size:]
        val_examples = examples[:val_size]
        
        print(f"📊 Données divisées: {len(train_examples)} entraînement, {len(val_examples)} validation")
        
        return train_examples, val_examples
    
    def evaluate_model(self, examples: List[Example]) -> Dict[str, float]:
        """
        Évalue les performances du modèle sur un ensemble de données
        
        Args:
            examples: Exemples à évaluer
            
        Returns:
            Dict: Métriques de performance (précision, rappel, F1-score)
        """
        if not examples:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        scores = self.nlp.evaluate(examples)
        
        # Extrait les métriques principales
        metrics = {
            "precision": scores.get("ents_p", 0.0),
            "recall": scores.get("ents_r", 0.0), 
            "f1": scores.get("ents_f", 0.0),
            "accuracy": scores.get("token_acc", 0.0)
        }
        
        return metrics
    
    def train_model(self, training_data: List[Tuple[str, Dict]], 
                   config: Dict[str, Any] = None,
                   progress_callback: Callable[[int, int, Dict], None] = None) -> Dict[str, Any]:
        """
        Effectue l'entraînement du modèle avec suivi des performances
        
        Args:
            training_data: Données d'entraînement brutes
            config: Configuration d'entraînement
            progress_callback: Fonction de callback pour le suivi
            
        Returns:
            Dict: Résultats d'entraînement et métriques finales
        """
        # Utilise la configuration par défaut si non fournie
        if config is None:
            config = self.default_config.copy()
        
        print(f"🚀 Début de l'entraînement avec {len(training_data)} exemples")
        print(f"⚙️ Configuration: {config}")
        
        try:
            # Prépare les données
            examples = self.prepare_training_data(training_data)
            if not examples:
                raise ValueError("Aucun exemple valide pour l'entraînement")
            
            # Divise les données
            train_examples, val_examples = self.split_data(
                examples, config.get('validation_split', 0.2)
            )
            
            # Initialise l'historique d'entraînement
            self.training_history = []
            best_f1 = 0.0
            patience_counter = 0
            best_model_state = None
            
            # Configuration des tailles de batch dynamiques
            batch_sizes = compounding(
                config.get('batch_size', 8), 
                config.get('batch_size', 8) * 2, 
                1.001
            )
            
            # Désactive les autres composants pendant l'entraînement
            other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
            with self.nlp.disable_pipes(*other_pipes):
                
                # Obtient l'optimizer
                optimizer = self.nlp.resume_training()
                
                print(f"🎯 Début de l'entraînement ({config.get('n_iter', 30)} époques)")
                
                for epoch in range(config.get('n_iter', 30)):
                    start_time = time.time()
                    
                    # Mélange les données d'entraînement
                    random.shuffle(train_examples)
                    
                    # Variables pour le suivi des pertes
                    losses = {}
                    batches_processed = 0
                    
                    # Entraînement par batch
                    for batch in minibatch(train_examples, size=next(batch_sizes)):
                        self.nlp.update(
                            batch, 
                            drop=config.get('dropout', 0.2),
                            losses=losses,
                            sgd=optimizer
                        )
                        batches_processed += 1
                    
                    # Évaluation sur les données de validation
                    train_metrics = self.evaluate_model(train_examples[:100])  # Sous-ensemble pour la vitesse
                    val_metrics = self.evaluate_model(val_examples)
                    
                    epoch_time = time.time() - start_time
                    
                    # Enregistre les métriques
                    epoch_info = {
                        'epoch': epoch + 1,
                        'train_loss': losses.get('ner', 0.0),
                        'train_precision': train_metrics['precision'],
                        'train_recall': train_metrics['recall'],
                        'train_f1': train_metrics['f1'],
                        'val_precision': val_metrics['precision'],
                        'val_recall': val_metrics['recall'],
                        'val_f1': val_metrics['f1'],
                        'epoch_time': epoch_time,
                        'batches_processed': batches_processed
                    }
                    
                    self.training_history.append(epoch_info)
                    
                    # Affichage des métriques
                    print(f"Époque {epoch + 1:2d}/{config.get('n_iter', 30)} | "
                          f"Loss: {losses.get('ner', 0.0):.4f} | "
                          f"Val F1: {val_metrics['f1']:.3f} | "
                          f"Temps: {epoch_time:.1f}s")
                    
                    # Callback de progression
                    if progress_callback:
                        progress_callback(epoch + 1, config.get('n_iter', 30), epoch_info)
                    
                    # Early stopping
                    current_f1 = val_metrics['f1']
                    if current_f1 > best_f1:
                        best_f1 = current_f1
                        patience_counter = 0
                        # Sauvegarde le meilleur état du modèle
                        best_model_state = self.nlp.to_bytes()
                        print(f"  🎉 Nouveau meilleur F1-score: {best_f1:.3f}")
                    else:
                        patience_counter += 1
                    
                    # Arrêt anticipé si pas d'amélioration
                    if patience_counter >= config.get('patience', 5) and epoch > 10:
                        print(f"🛑 Arrêt anticipé après {patience_counter} époques sans amélioration")
                        break
            
            # Restaure le meilleur modèle
            if best_model_state:
                self.nlp.from_bytes(best_model_state)
                print(f"✅ Meilleur modèle restauré (F1-score: {best_f1:.3f})")
            
            # Évaluation finale
            final_metrics = self.evaluate_model(val_examples)
            
            training_results = {
                'success': True,
                'epochs_completed': len(self.training_history),
                'best_f1_score': best_f1,
                'final_metrics': final_metrics,
                'training_history': self.training_history,
                'config_used': config
            }
            
            print(f"🏁 Entraînement terminé avec succès!")
            print(f"📊 Métriques finales: P={final_metrics['precision']:.3f}, "
                  f"R={final_metrics['recall']:.3f}, F1={final_metrics['f1']:.3f}")
            
            return training_results
            
        except Exception as e:
            print(f"❌ Erreur pendant l'entraînement: {e}")
            return {
                'success': False,
                'error': str(e),
                'training_history': self.training_history
            }
    
    def save_model(self, output_path: str = None, 
                   model_info: Dict[str, Any] = None) -> str:
        """
        Sauvegarde le modèle entraîné
        
        Args:
            output_path: Chemin de sauvegarde (généré automatiquement si None)
            model_info: Informations supplémentaires sur le modèle
            
        Returns:
            str: Chemin du modèle sauvegardé
        """
        if not self.nlp:
            raise ValueError("Aucun modèle à sauvegarder")
        
        # Génère un nom de fichier si non fourni
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"models/spacy_model_{timestamp}"
        
        # Crée le dossier de destination
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Sauvegarde le modèle SpaCy
            self.nlp.to_disk(output_dir)
            
            # Sauvegarde les métadonnées
            metadata = {
                'base_model': self.base_model_name,
                'custom_entities': self.custom_entities,
                'training_date': datetime.now().isoformat(),
                'training_history': self.training_history[-5:] if self.training_history else [],  # Dernières époques
                'model_info': model_info or {}
            }
            
            metadata_path = output_dir / "model_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Sauvegarde l'historique complet d'entraînement
            if self.training_history:
                history_path = output_dir / "training_history.json"
                with open(history_path, 'w', encoding='utf-8') as f:
                    json.dump(self.training_history, f, indent=2)
            
            print(f"💾 Modèle sauvegardé dans: {output_dir}")
            print(f"📋 Métadonnées sauvegardées: {metadata_path}")
            
            return str(output_dir)
            
        except Exception as e:
            raise Exception(f"Erreur lors de la sauvegarde: {e}")
    
    def load_trained_model(self, model_path: str) -> bool:
        """
        Charge un modèle précédemment entraîné
        
        Args:
            model_path: Chemin vers le modèle sauvegardé
            
        Returns:
            bool: True si le chargement a réussi
        """
        try:
            model_dir = Path(model_path)
            
            if not model_dir.exists():
                raise FileNotFoundError(f"Le modèle {model_path} n'existe pas")
            
            # Charge le modèle
            self.nlp = spacy.load(model_dir)
            
            # Charge les métadonnées si disponibles
            metadata_path = model_dir / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                self.base_model_name = metadata.get('base_model', 'unknown')
                self.custom_entities = metadata.get('custom_entities', [])
                
                print(f"📋 Métadonnées chargées: {len(self.custom_entities)} entités personnalisées")
            
            print(f"✅ Modèle chargé depuis: {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur le modèle actuel
        
        Returns:
            Dict: Informations détaillées sur le modèle
        """
        if not self.nlp:
            return {"status": "no_model_loaded"}
        
        info = {
            "base_model": self.base_model_name,
            "custom_entities": self.custom_entities,
            "pipeline_components": self.nlp.pipe_names,
            "vocab_size": len(self.nlp.vocab),
            "has_ner": "ner" in self.nlp.pipe_names,
            "training_epochs": len(self.training_history),
            "status": "ready"
        }
        
        if self.training_history:
            last_epoch = self.training_history[-1]
            info["last_performance"] = {
                "f1_score": last_epoch.get('val_f1', 0.0),
                "precision": last_epoch.get('val_precision', 0.0),
                "recall": last_epoch.get('val_recall', 0.0)
            }
        
        return info
    
    def test_model(self, test_text: str) -> Dict[str, Any]:
        """
        Teste le modèle sur un texte donné
        
        Args:
            test_text: Texte à analyser
            
        Returns:
            Dict: Résultats de l'analyse NER
        """
        if not self.nlp:
            return {"error": "Aucun modèle chargé"}
        
        try:
            doc = self.nlp(test_text)
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": getattr(ent, "_.confidence", 1.0)  # Si disponible
                })
            
            return {
                "text": test_text,
                "entities": entities,
                "entity_count": len(entities),
                "processed_successfully": True
            }
            
        except Exception as e:
            return {
                "error": f"Erreur lors du test: {e}",
                "processed_successfully": False
            }
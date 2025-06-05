# Fichier : modules/model_trainer.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module d'entraînement de modèles SpaCy pour NER personnalisé
===========================================================

Ce module gère le fine-tuning de modèles SpaCy existants.
Version corrigée et stabilisée.
"""

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Any
from datetime import datetime
import traceback

class SpacyModelTrainer:
    """
    Gestionnaire d'entraînement pour les modèles SpaCy NER personnalisés.
    """
    
    def __init__(self, base_model_name: str = "fr_core_news_sm"):
        """ Initialise le trainer avec un modèle de base. """
        self.base_model_name = base_model_name
        self.nlp = None
        self.ner = None
        self.training_history = []
        self.custom_entities = []
        self.default_config = {
            'n_iter': 30, 'dropout': 0.2, 'batch_size': 8,
            'patience': 5, 'validation_split': 0.2
        }
        print(f"🤖 Trainer initialisé avec le modèle de base : {base_model_name}")
    
    def load_base_model(self) -> bool:
        """ Charge le modèle SpaCy de base et prépare le composant NER. """
        try:
            print(f"📥 Chargement du modèle de base : {self.base_model_name}")
            self.nlp = spacy.load(self.base_model_name)
            
            if "ner" in self.nlp.pipe_names:
                self.ner = self.nlp.get_pipe("ner")
            else:
                self.ner = self.nlp.add_pipe("ner", last=True)
                
            print("✅ Modèle de base chargé et composant NER prêt.")
            return True
        except OSError:
            print(f"❌ ERREUR : Le modèle '{self.base_model_name}' n'est pas installé.")
            print(f"💡 Essayez : python -m spacy download {self.base_model_name}")
            return False
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle : {e}")
            return False
    
    def add_custom_entities(self, entity_types: List[str]) -> bool:
        """ Ajoute de nouvelles étiquettes d'entités au composant NER. """
        if not self.ner:
            print("❌ Le composant NER n'est pas initialisé. Chargez d'abord un modèle.")
            return False
        
        print(f"🏷️  Ajout des entités personnalisées : {entity_types}")
        for entity_type in entity_types:
            if entity_type not in self.ner.labels:
                self.ner.add_label(entity_type)
        
        self.custom_entities = list(self.ner.labels)
        print(f"🎯 Entités configurées dans le modèle : {self.custom_entities}")
        return True
    
    def prepare_training_data(self, raw_training_data: List[Tuple[str, Dict]]) -> List[Example]:
        """ Prépare les données brutes au format SpaCy Example. """
        examples = []
        print(f"🔄 Préparation de {len(raw_training_data)} exemples...")
        for text, annotations in raw_training_data:
            try:
                doc = self.nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            except Exception as e:
                print(f"⚠️ Erreur de préparation pour l'exemple '{text[:50]}...' : {e}")
        print(f"✅ {len(examples)} exemples préparés avec succès.")
        return examples
    
    def split_data(self, examples: List[Example], validation_split: float) -> Tuple[List[Example], List[Example]]:
        """ Divise les données en ensembles d'entraînement et de validation. """
        random.shuffle(examples)
        val_size = int(len(examples) * validation_split)
        train_examples = examples[val_size:]
        val_examples = examples[:val_size]
        print(f"📊 Données divisées : {len(train_examples)} entraînement, {len(val_examples)} validation.")
        return train_examples, val_examples
    
    def evaluate_model(self, examples: List[Example]) -> Dict[str, float]:
        """ Évalue les performances du modèle sur un ensemble de données. """
        if not examples:
            return {"ents_p": 0.0, "ents_r": 0.0, "ents_f": 0.0}
        return self.nlp.evaluate(examples)

    def train_model(self, training_data: List[Tuple[str, Dict]], 
                   config: Dict[str, Any] = None,
                   progress_callback: Callable[[int, int, Dict], None] = None) -> Dict[str, Any]:
        """
        Effectue l'entraînement du modèle avec la logique corrigée.
        """
        if config is None:
            config = self.default_config.copy()
        
        print(f"🚀 Début de l'entraînement avec la configuration : {config}")
        
        try:
            examples = self.prepare_training_data(training_data)
            if not examples:
                raise ValueError("Aucun exemple valide pour l'entraînement.")
            
            train_examples, val_examples = self.split_data(
                examples, config.get('validation_split', 0.2)
            )
            
            self.training_history = []
            best_f1_score = -1.0
            patience_counter = 0
            best_model_state = None
            
            other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
            with self.nlp.disable_pipes(*other_pipes):
                optimizer = self.nlp.resume_training()
                
                print(f"🎯 Boucle d'entraînement démarrée pour {config.get('n_iter', 30)} époques.")
                
                for epoch in range(config.get('n_iter', 30)):
                    start_time = time.time()
                    random.shuffle(train_examples)
                    losses = {}
                    
                    # Utilisation correcte du générateur de taille de lots
                    batch_sizes_generator = compounding(
                        config.get('batch_size', 4.0), 
                        32.0,
                        1.001
                    )
                    
                    for batch in minibatch(train_examples, size=batch_sizes_generator):
                        self.nlp.update(
                            batch, 
                            drop=config.get('dropout', 0.2),
                            losses=losses,
                            sgd=optimizer
                        )
                    
                    val_scores = self.evaluate_model(val_examples)
                    current_f1 = val_scores.get("ents_f", 0.0)
                    epoch_time = time.time() - start_time
                    
                    epoch_info = {
                        'epoch': epoch + 1,
                        'train_loss': losses.get('ner', 0.0),
                        'val_f1': current_f1,
                        'epoch_time': epoch_time
                    }
                    self.training_history.append(epoch_info)
                    
                    if progress_callback:
                        progress_callback(epoch + 1, config.get('n_iter', 30), epoch_info)
                    
                    # Early stopping pour éviter le sur-entraînement
                    if current_f1 > best_f1_score:
                        best_f1_score = current_f1
                        patience_counter = 0
                        best_model_state = self.nlp.to_bytes()
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= config.get('patience', 5) and epoch > 10:
                        print(f"🛑 Arrêt anticipé après {epoch + 1} époques.")
                        break
            
            if best_model_state:
                print(f"✅ Restauration du meilleur modèle (F1-score : {best_f1_score:.3f}).")
                self.nlp.from_bytes(best_model_state)
            
            final_metrics = self.evaluate_model(val_examples)
            
            return {
                'success': True,
                'epochs_completed': len(self.training_history),
                'final_metrics': {
                    "precision": final_metrics.get("ents_p", 0.0),
                    "recall": final_metrics.get("ents_r", 0.0),
                    "f1": final_metrics.get("ents_f", 0.0)
                }
            }
            
        except Exception as e:
            print(f"❌ Erreur majeure dans train_model : {e}")
            traceback.print_exc() # Imprime une trace détaillée de l'erreur dans la console
            return {'success': False, 'error': str(e)}

    def save_model(self, output_path: str, model_info: Dict[str, Any] = None) -> str:
        """ Sauvegarde le modèle entraîné sur le disque. """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.nlp.to_disk(output_dir)
        
        metadata = {
            'base_model': self.base_model_name,
            'custom_entities': self.custom_entities,
            'training_date': datetime.now().isoformat(),
            'model_info': model_info or {}
        }
        with open(output_dir / "model_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"💾 Modèle sauvegardé dans : {output_dir}")
        return str(output_dir)
    
    def load_trained_model(self, model_path: str) -> bool:
        """ Charge un modèle précédemment entraîné. """
        try:
            self.nlp = spacy.load(model_path)
            print(f"✅ Modèle chargé depuis : {model_path}")
            return True
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle depuis {model_path} : {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """ Retourne des informations sur le modèle actuellement chargé. """
        if not self.nlp:
            return {"status": "Aucun modèle chargé"}
        
        return {
            "base_model": self.base_model_name,
            "custom_entities": self.nlp.get_pipe("ner").labels,
            "pipeline_components": self.nlp.pipe_names,
        }
    
    def test_model(self, test_text: str) -> Dict[str, Any]:
        """ Teste le modèle sur un texte donné. """
        if not self.nlp:
            return {"error": "Aucun modèle chargé.", "processed_successfully": False}
        
        try:
            doc = self.nlp(test_text)
            entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
            return {"entities": entities, "processed_successfully": True}
        except Exception as e:
            return {"error": str(e), "processed_successfully": False}
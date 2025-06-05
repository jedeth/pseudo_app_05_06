#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Générateur de données d'entraînement pour SpaCy
==============================================

Ce module génère automatiquement des données d'entraînement au format SpaCy
à partir de listes de termes fournis par l'utilisateur. Il crée des phrases
contextuelles réalistes pour chaque type d'entité NER personnalisée.
"""

import random
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any
from datetime import datetime

class TrainingDataGenerator:
    """
    Générateur automatique de données d'entraînement pour SpaCy NER
    
    Cette classe prend en entrée des listes de termes pour chaque entité
    personnalisée et génère des phrases d'entraînement contextuelles
    avec les annotations au format SpaCy.
    """
    
    def __init__(self):
        """
        Initialise le générateur avec des modèles de phrases prédéfinis
        pour différents types d'entités courantes
        """
        # Modèles de phrases par type d'entité
        # Chaque template contient {entity} qui sera remplacé par le terme réel
        self.sentence_templates = {
            # Templates pour les établissements (codes, identifiants)
            'ETABLISSEMENT': [
                "L'établissement {entity} a été créé en 2020.",
                "Le code d'établissement {entity} correspond à notre filiale principale.",
                "Nous travaillons avec l'établissement référencé {entity}.",
                "L'identifiant {entity} désigne un établissement certifié.",
                "Le site {entity} emploie plus de 50 personnes.",
                "L'établissement {entity} respecte toutes les normes de qualité.",
                "Notre partenaire {entity} livre régulièrement nos produits.",
                "L'audit de l'établissement {entity} aura lieu le mois prochain.",
                "Le code {entity} figure dans notre base de données clients.",
                "L'établissement {entity} bénéficie d'une accréditation spéciale."
            ],
            
            # Templates pour les personnes (noms, prénoms, identifiants)
            'PERSONNE': [
                "Monsieur {entity} travaille dans ce service depuis 5 ans.",
                "Le dossier de {entity} nécessite une révision urgente.",
                "Madame {entity} sera responsable de ce projet.",
                "{entity} a participé à la réunion d'hier.",
                "Le rapport de {entity} sera présenté demain.",
                "Nous devons contacter {entity} rapidement.",
                "{entity} est expert dans son domaine.",
                "L'employé {entity} prend ses vacances en août.",
                "Le Dr {entity} consulte tous les mardis.",
                "{entity} a signé le contrat ce matin."
            ],
            
            # Templates pour les organisations
            'ORGANISATION': [
                "L'organisation {entity} publie un rapport annuel.",
                "Nous collaborons avec {entity} depuis plusieurs années.",
                "{entity} organise une conférence internationale.",
                "Le partenariat avec {entity} est très fructueux.",
                "{entity} a lancé une nouvelle initiative.",
                "Les membres de {entity} se réunissent chaque mois.",
                "{entity} contribue activement au développement local.",
                "Le président de {entity} donnera une interview.",
                "{entity} recrute de nouveaux collaborateurs.",
                "Les statuts de {entity} ont été modifiés récemment."
            ],
            
            # Templates pour les lieux
            'LIEU': [
                "La réunion aura lieu à {entity}.",
                "Nous nous rendons souvent à {entity}.",
                "{entity} est un endroit magnifique.",
                "Le siège social se trouve à {entity}.",
                "Les formations ont lieu à {entity}.",
                "{entity} accueille de nombreux visiteurs.",
                "L'événement se déroulera à {entity}.",
                "Nous livrons régulièrement à {entity}.",
                "{entity} dispose d'excellentes infrastructures.",
                "Le bureau de {entity} ferme à 18h."
            ],
            
            # Templates pour les codes/identifiants génériques
            'CODE': [
                "Le code {entity} est utilisé pour identifier ce produit.",
                "Veuillez saisir le code {entity} dans le système.",
                "Le référence {entity} correspond à cet article.",
                "L'identifiant {entity} permet de tracer l'opération.",
                "Le numéro {entity} figure sur la facture.",
                "Utilisez le code {entity} pour accéder au service.",
                "Le système affiche l'erreur {entity}.",
                "La commande {entity} a été expédiée hier.",
                "Le ticket {entity} est en cours de traitement.",
                "La référence {entity} n'existe pas dans notre base."
            ]
        }
        
        # Mots de liaison et connecteurs pour enrichir les phrases
        self.connectors = [
            "En effet,", "Par ailleurs,", "De plus,", "Cependant,", "Néanmoins,",
            "Ainsi,", "Par conséquent,", "En outre,", "D'autre part,", "Finalement,"
        ]
        
        # Compléments contextuels
        self.context_additions = [
            "conformément à la réglementation",
            "selon nos procédures internes",
            "dans le cadre de notre activité",
            "pour améliorer nos services",
            "afin de répondre aux besoins",
            "en respect des normes en vigueur"
        ]
        
    def load_terms_from_file(self, filepath: str) -> List[str]:
        """
        Charge une liste de termes depuis un fichier texte
        
        Args:
            filepath (str): Chemin vers le fichier contenant les termes (un par ligne)
            
        Returns:
            List[str]: Liste des termes chargés et nettoyés
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            Exception: Pour toute autre erreur de lecture
        """
        try:
            terms = []
            with open(filepath, 'r', encoding='utf-8') as file:
                for line in file:
                    # Nettoie chaque ligne (supprime espaces et caractères spéciaux)
                    term = line.strip()
                    if term and not term.startswith('#'):  # Ignore les lignes vides et commentaires
                        terms.append(term)
            
            print(f"✅ {len(terms)} termes chargés depuis {filepath}")
            return terms
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Le fichier {filepath} n'a pas été trouvé")
        except Exception as e:
            raise Exception(f"Erreur lors de la lecture du fichier {filepath}: {str(e)}")
    
    def generate_sentence_for_term(self, term: str, entity_type: str, 
                                  add_context: bool = True) -> str:
        """
        Génère une phrase contextualisée pour un terme donné
        
        Args:
            term (str): Le terme à intégrer dans la phrase
            entity_type (str): Type d'entité (PERSONNE, ETABLISSEMENT, etc.)
            add_context (bool): Ajouter du contexte supplémentaire
            
        Returns:
            str: Phrase générée contenant le terme
        """
        # Sélectionne un template approprié ou utilise un template générique
        if entity_type in self.sentence_templates:
            templates = self.sentence_templates[entity_type]
        else:
            # Template générique si le type d'entité n'est pas reconnu
            templates = [
                "Le terme {entity} apparaît dans ce document.",
                "Nous devons traiter {entity} avec attention.",
                "L'élément {entity} est important pour cette analyse.",
                "Il faut considérer {entity} dans notre évaluation.",
                "Le cas {entity} nécessite un examen approfondi."
            ]
        
        # Sélectionne un template au hasard
        template = random.choice(templates)
        
        # Remplace le placeholder par le terme réel
        sentence = template.replace("{entity}", term)
        
        # Ajoute parfois du contexte supplémentaire pour enrichir
        if add_context and random.random() < 0.3:  # 30% de chance
            connector = random.choice(self.connectors)
            context = random.choice(self.context_additions)
            sentence = f"{connector} {sentence.lower()} {context}."
        
        return sentence
    
    def create_spacy_annotation(self, sentence: str, term: str, 
                              entity_type: str) -> Tuple[str, Dict[str, List]]:
        """
        Crée une annotation au format SpaCy pour une phrase donnée
        
        Args:
            sentence (str): Phrase contenant l'entité
            term (str): Terme à annoter
            entity_type (str): Type d'entité NER
            
        Returns:
            Tuple[str, Dict]: Phrase et dictionnaire d'annotations SpaCy
        """
        # Recherche toutes les occurrences du terme dans la phrase
        entities = []
        
        # Utilise une recherche insensible à la casse
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        
        for match in pattern.finditer(sentence):
            start = match.start()
            end = match.end()
            
            # Ajoute l'annotation de l'entité
            entities.append((start, end, entity_type))
        
        # Format d'annotation SpaCy
        annotations = {
            "entities": entities
        }
        
        return sentence, annotations
    
    def generate_training_data(self, entity_files: Dict[str, str], 
                             sentences_per_term: int = 5,
                             add_variations: bool = True) -> List[Tuple[str, Dict]]:
        """
        Génère un ensemble complet de données d'entraînement
        
        Args:
            entity_files (Dict[str, str]): Dictionnaire {type_entité: chemin_fichier}
            sentences_per_term (int): Nombre de phrases à générer par terme
            add_variations (bool): Ajouter des variations contextuelles
            
        Returns:
            List[Tuple[str, Dict]]: Données d'entraînement au format SpaCy
        """
        training_data = []
        generation_stats = {}
        
        print("🚀 Début de la génération des données d'entraînement...")
        
        for entity_type, filepath in entity_files.items():
            print(f"\n📂 Traitement des entités de type '{entity_type}'...")
            
            try:
                # Charge les termes depuis le fichier
                terms = self.load_terms_from_file(filepath)
                generated_count = 0
                
                for term in terms:
                    # Génère plusieurs phrases pour chaque terme
                    for i in range(sentences_per_term):
                        try:
                            # Génère une phrase
                            sentence = self.generate_sentence_for_term(
                                term, entity_type, add_variations
                            )
                            
                            # Crée l'annotation SpaCy
                            annotated_sentence, annotations = self.create_spacy_annotation(
                                sentence, term, entity_type
                            )
                            
                            # Ajoute aux données d'entraînement
                            training_data.append((annotated_sentence, annotations))
                            generated_count += 1
                            
                        except Exception as e:
                            print(f"⚠️ Erreur lors de la génération pour '{term}': {e}")
                            continue
                
                generation_stats[entity_type] = {
                    'terms_count': len(terms),
                    'sentences_generated': generated_count
                }
                
                print(f"✅ {generated_count} phrases générées pour {len(terms)} termes de type '{entity_type}'")
                
            except Exception as e:
                print(f"❌ Erreur lors du traitement de '{entity_type}': {e}")
                generation_stats[entity_type] = {'error': str(e)}
                continue
        
        # Mélange les données pour un meilleur entraînement
        random.shuffle(training_data)
        
        print(f"\n🎯 Génération terminée: {len(training_data)} phrases d'entraînement créées")
        
        return training_data, generation_stats
    
    def save_training_data(self, training_data: List[Tuple[str, Dict]], 
                      filename: str = None) -> str:
    # """
    # Sauvegarde les données d'entraînement au format JSON
    
    # Args:
    #     training_data: Données d'entraînement générées
    #     filename: Nom du fichier (généré automatiquement si None)
        
    # Returns:
    #     str: Chemin du fichier sauvegardé
    # """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}.json"
    
    # Crée le dossier data s'il n'existe pas
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
    
        filepath = data_dir / filename
    
        try:
        # Convertit les données au format JSON standard
            json_data = []
            for text, annotations in training_data:
                # Vérifie le format des données
                if not isinstance(text, str):
                    print(f"⚠️ Texte invalide ignoré: {type(text)}")
                    continue
                
                if not isinstance(annotations, dict) or 'entities' not in annotations:
                    print(f"⚠️ Annotations invalides ignorées: {annotations}")
                    continue
            
                json_data.append({
                    "text": text,
                    "entities": annotations["entities"]
                })
        
            if not json_data:
                raise ValueError("Aucune donnée valide à sauvegarder")
        
            # Sauvegarde au format JSON standard
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
        
            print(f"💾 {len(json_data)} exemples sauvegardés dans: {filepath}")
            return str(filepath)
        
        except Exception as e:
            raise Exception(f"Erreur lors de la sauvegarde: {e}")
    
    def preview_training_data(self, training_data: List[Tuple[str, Dict]], 
                            max_examples: int = 10) -> str:
        """
        Génère un aperçu des données d'entraînement pour prévisualisation
        
        Args:
            training_data: Données d'entraînement
            max_examples: Nombre maximum d'exemples à afficher
            
        Returns:
            str: Texte de prévisualisation formaté
        """
        if not training_data:
            return "Aucune donnée d'entraînement disponible."
        
        preview_text = f"📊 APERÇU DES DONNÉES D'ENTRAÎNEMENT\n"
        preview_text += f"{'='*50}\n\n"
        preview_text += f"Total d'exemples générés: {len(training_data)}\n\n"
        
        # Affiche quelques exemples
        preview_text += f"🔍 EXEMPLES ({min(max_examples, len(training_data))} premiers):\n"
        preview_text += f"{'-'*40}\n\n"
        
        for i, (sentence, annotations) in enumerate(training_data[:max_examples]):
            preview_text += f"Exemple {i+1}:\n"
            preview_text += f"Phrase: {sentence}\n"
            
            if annotations["entities"]:
                preview_text += "Entités détectées:\n"
                for start, end, label in annotations["entities"]:
                    entity_text = sentence[start:end]
                    preview_text += f"  - '{entity_text}' ({label}) [position {start}-{end}]\n"
            else:
                preview_text += "Aucune entité détectée\n"
            
            preview_text += "\n"
        
        # Statistiques par type d'entité
        entity_stats = {}
        for sentence, annotations in training_data:
            for start, end, label in annotations["entities"]:
                entity_stats[label] = entity_stats.get(label, 0) + 1
        
        if entity_stats:
            preview_text += f"📈 STATISTIQUES PAR TYPE D'ENTITÉ:\n"
            preview_text += f"{'-'*30}\n"
            for entity_type, count in sorted(entity_stats.items()):
                preview_text += f"{entity_type}: {count} occurrences\n"
        
        return preview_text
    
    def validate_training_data(self, training_data: List[Tuple[str, Dict]]) -> Dict[str, Any]:
        """
        Valide la qualité des données d'entraînement générées
        
        Args:
            training_data: Données à valider
            
        Returns:
            Dict: Rapport de validation avec statistiques et erreurs
        """
        validation_report = {
            'total_examples': len(training_data),
            'valid_examples': 0,
            'invalid_examples': 0,
            'errors': [],
            'entity_distribution': {},
            'sentence_length_stats': {
                'min': float('inf'),
                'max': 0,
                'avg': 0
            }
        }
        
        sentence_lengths = []
        
        for i, (sentence, annotations) in enumerate(training_data):
            is_valid = True
            sentence_length = len(sentence.split())
            sentence_lengths.append(sentence_length)
            
            # Vérifie que les annotations sont cohérentes
            for start, end, label in annotations.get("entities", []):
                # Vérifie que les positions sont valides
                if start >= end or end > len(sentence):
                    validation_report['errors'].append(
                        f"Exemple {i}: Position d'entité invalide ({start}-{end})"
                    )
                    is_valid = False
                
                # Compte les entités par type
                validation_report['entity_distribution'][label] = \
                    validation_report['entity_distribution'].get(label, 0) + 1
            
            if is_valid:
                validation_report['valid_examples'] += 1
            else:
                validation_report['invalid_examples'] += 1
        
        # Calcule les statistiques de longueur
        if sentence_lengths:
            validation_report['sentence_length_stats']['min'] = min(sentence_lengths)
            validation_report['sentence_length_stats']['max'] = max(sentence_lengths)
            validation_report['sentence_length_stats']['avg'] = sum(sentence_lengths) / len(sentence_lengths)
        
        return validation_report
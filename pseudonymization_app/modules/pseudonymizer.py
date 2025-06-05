#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de pseudonymisation et dépseudonymisation de textes
========================================================

Ce module utilise des modèles SpaCy entraînés pour identifier et remplacer
des entités sensibles dans des textes, tout en maintenant un fichier de
correspondance pour permettre la dépseudonymisation.
"""

import spacy
import json
import hashlib
import random
import string
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import uuid

class TextPseudonymizer:
    """
    Gestionnaire de pseudonymisation et dépseudonymisation de textes
    
    Cette classe permet de :
    - Charger un modèle SpaCy entraîné
    - Identifier les entités sensibles dans un texte
    - Remplacer ces entités par des pseudonymes
    - Maintenir un fichier de correspondance
    - Restaurer le texte original (dépseudonymisation)
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialise le pseudonymiseur
        
        Args:
            model_path (str): Chemin vers le modèle SpaCy entraîné
        """
        self.nlp = None
        self.model_path = model_path
        self.correspondence_map = {}  # {pseudonyme: entité_originale}
        self.reverse_map = {}  # {entité_originale: pseudonyme}
        self.entity_counters = {}  # Compteurs pour générer des pseudonymes uniques
        
        # Stratégies de pseudonymisation par type d'entité
        self.pseudonym_strategies = {
            'PERSONNE': {
                'type': 'structured',
                'prefix': 'PERS',
                'format': 'PERS_{counter:04d}',
                'alternatives': ['Jean Dupont', 'Marie Martin', 'Pierre Durand', 'Sophie Bernard']
            },
            'ETABLISSEMENT': {
                'type': 'structured', 
                'prefix': 'ETAB',
                'format': 'ETAB_{counter:04d}',
                'alternatives': []
            },
            'ORGANISATION': {
                'type': 'structured',
                'prefix': 'ORG',
                'format': 'ORG_{counter:04d}',
                'alternatives': ['ACME Corp', 'Société Alpha', 'Entreprise Beta']
            },
            'LIEU': {
                'type': 'structured',
                'prefix': 'LIEU',
                'format': 'LIEU_{counter:04d}',
                'alternatives': ['Paris', 'Lyon', 'Marseille', 'Toulouse']
            },
            'CODE': {
                'type': 'hash',
                'prefix': 'CODE',
                'format': 'CODE_{hash}',
                'alternatives': []
            }
        }
        
        # Chargement automatique du modèle si fourni
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Charge un modèle SpaCy depuis un chemin donné
        
        Args:
            model_path (str): Chemin vers le modèle
            
        Returns:
            bool: True si le chargement a réussi
        """
        try:
            print(f"📥 Chargement du modèle depuis: {model_path}")
            
            # Charge le modèle SpaCy
            self.nlp = spacy.load(model_path)
            self.model_path = model_path
            
            # Vérifie que le composant NER est présent
            if "ner" not in self.nlp.pipe_names:
                print("⚠️ Attention: Aucun composant NER trouvé dans le modèle")
                return False
            
            print("✅ Modèle chargé avec succès")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            return False
    
    def generate_pseudonym(self, original_entity: str, entity_type: str, 
                          strategy: str = 'structured') -> str:
        """
        Génère un pseudonyme pour une entité donnée
        
        Args:
            original_entity (str): Entité originale
            entity_type (str): Type d'entité (PERSONNE, ETABLISSEMENT, etc.)
            strategy (str): Stratégie de pseudonymisation
            
        Returns:
            str: Pseudonyme généré
        """
        # Vérifie si un pseudonyme existe déjà pour cette entité
        if original_entity in self.reverse_map:
            return self.reverse_map[original_entity]
        
        # Obtient la configuration pour ce type d'entité
        config = self.pseudonym_strategies.get(entity_type, {
            'type': 'structured',
            'prefix': 'ENT',
            'format': 'ENT_{counter:04d}',
            'alternatives': []
        })
        
        pseudonym = None
        
        if config['type'] == 'structured':
            # Génération structurée avec compteur
            if entity_type not in self.entity_counters:
                self.entity_counters[entity_type] = 1
            else:
                self.entity_counters[entity_type] += 1
            
            pseudonym = config['format'].format(
                counter=self.entity_counters[entity_type],
                prefix=config['prefix']
            )
            
        elif config['type'] == 'hash':
            # Génération basée sur un hash
            hash_object = hashlib.md5(original_entity.encode())
            hash_hex = hash_object.hexdigest()[:8]  # 8 premiers caractères
            pseudonym = config['format'].format(hash=hash_hex.upper())
            
        elif config['type'] == 'alternative' and config['alternatives']:
            # Sélection d'une alternative prédéfinie
            pseudonym = random.choice(config['alternatives'])
            
        elif config['type'] == 'random':
            # Génération aléatoire
            random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            pseudonym = f"{config['prefix']}_{random_str}"
        
        # Par défaut, utilise la stratégie structurée
        if not pseudonym:
            if entity_type not in self.entity_counters:
                self.entity_counters[entity_type] = 1
            else:
                self.entity_counters[entity_type] += 1
            
            pseudonym = f"{config['prefix']}_{self.entity_counters[entity_type]:04d}"
        
        # Assure l'unicité du pseudonyme
        base_pseudonym = pseudonym
        counter = 1
        while pseudonym in self.correspondence_map:
            pseudonym = f"{base_pseudonym}_{counter}"
            counter += 1
        
        # Enregistre la correspondance
        self.correspondence_map[pseudonym] = original_entity
        self.reverse_map[original_entity] = pseudonym
        
        return pseudonym
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrait les entités d'un texte avec le modèle NER
        
        Args:
            text (str): Texte à analyser
            
        Returns:
            List[Dict]: Liste des entités détectées avec leurs informations
        """
        if not self.nlp:
            raise ValueError("Aucun modèle chargé")
        
        # Analyse du texte avec SpaCy
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': getattr(ent, '_.confidence', 1.0)  # Si disponible
            }
            entities.append(entity_info)
        
        # Trie les entités par position (important pour le remplacement)
        entities.sort(key=lambda x: x['start'], reverse=True)
        
        return entities
    
    def pseudonymize_text(self, text: str, 
                         entity_types_to_mask: List[str] = None,
                         preserve_format: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Pseudonymise un texte en remplaçant les entités identifiées
        
        Args:
            text (str): Texte à pseudonymiser
            entity_types_to_mask (List[str]): Types d'entités à masquer (None = tous)
            preserve_format (bool): Préserver le formatage du texte
            
        Returns:
            Tuple[str, Dict]: (texte pseudonymisé, informations de pseudonymisation)
        """
        if not self.nlp:
            raise ValueError("Aucun modèle chargé")
        
        print(f"🔒 Pseudonymisation du texte ({len(text)} caractères)...")
        
        # Extrait les entités
        entities = self.extract_entities(text)
        
        # Filtre les entités selon les types demandés
        if entity_types_to_mask:
            entities = [ent for ent in entities if ent['label'] in entity_types_to_mask]
        
        print(f"🎯 {len(entities)} entités détectées pour pseudonymisation")
        
        # Statistiques de pseudonymisation
        pseudonymization_stats = {
            'original_length': len(text),
            'entities_processed': 0,
            'entities_by_type': {},
            'pseudonyms_created': 0,
            'pseudonyms_reused': 0
        }
        
        # Copie du texte pour modification
        pseudonymized_text = text
        
        # Remplace les entités (en ordre inverse pour préserver les positions)
        for entity in entities:
            original_text = entity['text']
            entity_type = entity['label']
            start_pos = entity['start']
            end_pos = entity['end']
            
            # Vérifie si un pseudonyme existe déjà
            if original_text in self.reverse_map:
                pseudonym = self.reverse_map[original_text]
                pseudonymization_stats['pseudonyms_reused'] += 1
            else:
                pseudonym = self.generate_pseudonym(original_text, entity_type)
                pseudonymization_stats['pseudonyms_created'] += 1
            
            # Préservation du format (majuscules, casse, etc.)
            if preserve_format and original_text:
                if original_text.isupper():
                    pseudonym = pseudonym.upper()
                elif original_text.istitle():
                    pseudonym = pseudonym.title()
                elif original_text.islower():
                    pseudonym = pseudonym.lower()
            
            # Remplacement dans le texte
            pseudonymized_text = (pseudonymized_text[:start_pos] + 
                                pseudonym + 
                                pseudonymized_text[end_pos:])
            
            # Met à jour les statistiques
            pseudonymization_stats['entities_processed'] += 1
            if entity_type not in pseudonymization_stats['entities_by_type']:
                pseudonymization_stats['entities_by_type'][entity_type] = 0
            pseudonymization_stats['entities_by_type'][entity_type] += 1
        
        pseudonymization_stats['final_length'] = len(pseudonymized_text)
        
        print(f"✅ Pseudonymisation terminée: {pseudonymization_stats['entities_processed']} entités traitées")
        
        return pseudonymized_text, pseudonymization_stats
    
    def depseudonymize_text(self, pseudonymized_text: str, 
                           correspondence_map: Dict[str, str] = None) -> str:
        """
        Restaure un texte pseudonymisé vers sa forme originale
        
        Args:
            pseudonymized_text (str): Texte pseudonymisé
            correspondence_map (Dict): Correspondances (si différente de l'interne)
            
        Returns:
            str: Texte original restauré
        """
        print(f"🔓 Dépseudonymisation du texte ({len(pseudonymized_text)} caractères)...")
        
        # Utilise la carte fournie ou la carte interne
        corresp_map = correspondence_map or self.correspondence_map
        
        if not corresp_map:
            raise ValueError("Aucune correspondance disponible pour la dépseudonymisation")
        
        # Copie du texte pour modification
        depseudonymized_text = pseudonymized_text
        
        # Compte les remplacements effectués
        replacements_made = 0
        
        # Trie les pseudonymes par longueur décroissante pour éviter les remplacements partiels
        sorted_pseudonyms = sorted(corresp_map.keys(), key=len, reverse=True)
        
        for pseudonym in sorted_pseudonyms:
            original_entity = corresp_map[pseudonym]
            
            # Compte les occurrences avant remplacement
            occurrences_before = depseudonymized_text.count(pseudonym)
            
            if occurrences_before > 0:
                # Remplace toutes les occurrences du pseudonyme
                depseudonymized_text = depseudonymized_text.replace(pseudonym, original_entity)
                replacements_made += occurrences_before
        
        print(f"✅ Dépseudonymisation terminée: {replacements_made} remplacements effectués")
        
        return depseudonymized_text
    
    def save_correspondence_file(self, filepath: str = None, 
                               additional_info: Dict[str, Any] = None) -> str:
        """
        Sauvegarde le fichier de correspondance pour la dépseudonymisation
        
        Args:
            filepath (str): Chemin de sauvegarde (généré automatiquement si None)
            additional_info (Dict): Informations supplémentaires à inclure
            
        Returns:
            str: Chemin du fichier sauvegardé
        """
        if not self.correspondence_map:
            raise ValueError("Aucune correspondance à sauvegarder")
        
        # Génère un nom de fichier si non fourni
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = str(uuid.uuid4())[:8]
            filepath = f"correspondance_{timestamp}_{session_id}.json"
        
        # Prépare les données à sauvegarder
        correspondence_data = {
            'metadata': {
                'creation_date': datetime.now().isoformat(),
                'model_used': self.model_path,
                'total_pseudonyms': len(self.correspondence_map),
                'entity_types': list(self.entity_counters.keys()),
                'session_id': str(uuid.uuid4())
            },
            'correspondences': self.correspondence_map,
            'entity_counters': self.entity_counters,
            'additional_info': additional_info or {}
        }
        
        try:
            # Sauvegarde le fichier JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(correspondence_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Fichier de correspondance sauvegardé: {filepath}")
            return filepath
            
        except Exception as e:
            raise Exception(f"Erreur lors de la sauvegarde: {e}")
    
    def load_correspondence_file(self, filepath: str) -> bool:
        """
        Charge un fichier de correspondance pour la dépseudonymisation
        
        Args:
            filepath (str): Chemin vers le fichier de correspondance
            
        Returns:
            bool: True si le chargement a réussi
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                correspondence_data = json.load(f)
            
            # Charge les correspondances
            self.correspondence_map = correspondence_data.get('correspondences', {})
            self.entity_counters = correspondence_data.get('entity_counters', {})
            
            # Reconstruit la carte inverse
            self.reverse_map = {v: k for k, v in self.correspondence_map.items()}
            
            metadata = correspondence_data.get('metadata', {})
            print(f"📥 Correspondances chargées: {metadata.get('total_pseudonyms', 0)} pseudonymes")
            print(f"📅 Créé le: {metadata.get('creation_date', 'Date inconnue')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement des correspondances: {e}")
            return False
    
    def get_pseudonymization_summary(self) -> Dict[str, Any]:
        """
        Retourne un résumé de l'état actuel de pseudonymisation
        
        Returns:
            Dict: Résumé avec statistiques et informations
        """
        summary = {
            'model_loaded': self.nlp is not None,
            'model_path': self.model_path,
            'total_pseudonyms': len(self.correspondence_map),
            'entity_types_processed': list(self.entity_counters.keys()),
            'entity_counters': self.entity_counters.copy(),
            'available_strategies': list(self.pseudonym_strategies.keys())
        }
        
        # Statistiques par type d'entité
        entity_stats = {}
        for pseudonym, original in self.correspondence_map.items():
            # Détermine le type d'entité à partir du préfixe du pseudonyme
            entity_type = 'UNKNOWN'
            for etype in self.pseudonym_strategies.keys():
                prefix = self.pseudonym_strategies[etype]['prefix']
                if pseudonym.startswith(prefix):
                    entity_type = etype
                    break
            
            if entity_type not in entity_stats:
                entity_stats[entity_type] = 0
            entity_stats[entity_type] += 1
        
        summary['pseudonyms_by_type'] = entity_stats
        
        return summary
    
    def reset_correspondences(self):
        """
        Remet à zéro toutes les correspondances
        """
        self.correspondence_map.clear()
        self.reverse_map.clear()
        self.entity_counters.clear()
        print("🔄 Correspondances remises à zéro")
    
    def preview_pseudonymization(self, text: str, 
                                entity_types_to_mask: List[str] = None) -> Dict[str, Any]:
        """
        Prévisualise la pseudonymisation sans effectuer les remplacements
        
        Args:
            text (str): Texte à analyser
            entity_types_to_mask (List[str]): Types d'entités à considérer
            
        Returns:
            Dict: Aperçu des entités qui seraient pseudonymisées
        """
        if not self.nlp:
            raise ValueError("Aucun modèle chargé")
        
        entities = self.extract_entities(text)
        
        if entity_types_to_mask:
            entities = [ent for ent in entities if ent['label'] in entity_types_to_mask]
        
        preview = {
            'total_entities': len(entities),
            'entities_by_type': {},
            'entities_details': [],
            'would_create_pseudonyms': 0,
            'would_reuse_pseudonyms': 0
        }
        
        for entity in entities:
            entity_type = entity['label']
            original_text = entity['text']
            
            # Compte par type
            if entity_type not in preview['entities_by_type']:
                preview['entities_by_type'][entity_type] = 0
            preview['entities_by_type'][entity_type] += 1
            
            # Détermine si un nouveau pseudonyme serait créé
            if original_text in self.reverse_map:
                pseudonym = self.reverse_map[original_text]
                preview['would_reuse_pseudonyms'] += 1
            else:
                pseudonym = f"[NOUVEAU_{entity_type}]"
                preview['would_create_pseudonyms'] += 1
            
            preview['entities_details'].append({
                'original': original_text,
                'type': entity_type,
                'position': f"{entity['start']}-{entity['end']}",
                'pseudonym': pseudonym,
                'is_new': original_text not in self.reverse_map
            })
        
        return preview
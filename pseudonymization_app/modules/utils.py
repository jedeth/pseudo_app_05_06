#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitaires pour l'application de pseudonymisation
=================================================

Ce module contient des fonctions utilitaires communes
utilisées dans toute l'application.
"""

import json
import os
import datetime
from pathlib import Path

class AppUtils:
    """
    Classe contenant diverses fonctions utilitaires
    pour l'application de pseudonymisation
    """
    
    @staticmethod
    def save_config(config_data, filename="app_config.json"):
        """
        Sauvegarde la configuration de l'application
        
        Args:
            config_data (dict): Données de configuration à sauvegarder
            filename (str): Nom du fichier de configuration
        """
        config_path = Path("config") / filename
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de la configuration: {e}")
            return False
    
    @staticmethod
    def load_config(filename="app_config.json"):
        """
        Charge la configuration de l'application
        
        Args:
            filename (str): Nom du fichier de configuration
            
        Returns:
            dict: Données de configuration ou dictionnaire vide si erreur
        """
        config_path = Path("config") / filename
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Erreur lors du chargement de la configuration: {e}")
        return {}
    
    @staticmethod
    def get_timestamp():
        """
        Retourne un timestamp formaté pour les noms de fichiers
        
        Returns:
            str: Timestamp au format YYYYMMDD_HHMMSS
        """
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def validate_file_extension(filepath, allowed_extensions):
        """
        Valide l'extension d'un fichier
        
        Args:
            filepath (str): Chemin vers le fichier
            allowed_extensions (list): Liste des extensions autorisées
            
        Returns:
            bool: True si l'extension est valide, False sinon
        """
        file_ext = Path(filepath).suffix.lower()
        return file_ext in [ext.lower() for ext in allowed_extensions]
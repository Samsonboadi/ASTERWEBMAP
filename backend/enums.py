# backend/enums.py
"""
Enumeration definitions for ASTER processing
"""

from enum import Enum

class AlterationIndices(Enum):
    """Alteration mapping indices"""
    ADVANCED_ARGILLIC = "advanced_argillic"
    ARGILLIC = "argillic"
    PROPYLITIC = "propylitic"
    PHYLLIC = "phyllic"
    SILICIFICATION = "silicification"
    IRON_ALTERATION = "iron_alteration"
    GOSSAN = "gossan"
    
    # New alteration types
    SULFIDE_ALTERATION = "sulfide_alteration"
    CHLORITE_EPIDOTE = "chlorite_epidote"
    SERICITE_PYRITE = "sericite_pyrite"
    CARBONATE_ALTERATION = "carbonate_alteration"

class GeologicalFeatures(Enum):
    """Geological feature types"""
    LINEAMENTS = "lineaments"
    FAULTS = "faults"
    CONTACTS = "contacts"
    FRACTURES = "fractures"
    FOLDS = "folds"

class BandCombinations(Enum):
    """Enhanced band combinations for geological mapping"""
    LITHOLOGICAL = "lithological"
    GENERAL_ALTERATION = "general_alteration"
    IRON_OXIDE = "iron_oxide"
    ALOH_MINERALS = "aloh_minerals"
    MGOH_CARBONATE = "mgoh_carbonate"
    CROSTA = "crosta"
    SULFIDE = "sulfide"
    CHLORITE_ALTERATION = "chlorite_alteration"

class BandRatioMaps(Enum):
    """Specific band ratio maps"""
    FERRIC_IRON = "ferric_iron"
    ALOH_CONTENT = "aloh_content"
    MGOH_CONTENT = "mgoh_content"

class GoldPathfinderIndices(Enum):
    """Enhanced mineral indices for gold exploration"""
    GOLD_ALTERATION = "gold_alteration"
    QUARTZ_ADULARIA = "quartz_adularia"
    PYRITE = "pyrite"
    ARSENOPYRITE = "arsenopyrite"
    SILICA = "silica"
    PROPYLITIC_GOLD = "propylitic_gold"
    ARGILLIC_GOLD = "argillic_gold"
    ADVANCED_ARGILLIC_GOLD = "advanced_argillic_gold"

class ProcessingStages(Enum):
    """Processing stages for ASTER data"""
    EXTRACT = "extract"
    MINERAL_MAPPING = "mineral_mapping"
    ALTERATION_MAPPING = "alteration_mapping"
    GEOLOGICAL_MAPPING = "geological_mapping"
    GOLD_PATHFINDER = "gold_pathfinder"
    ADVANCED_ANALYSIS = "advanced_analysis"

class ProcessingStatus(Enum):
    """Processing status for ASTER data"""
    IDLE = "idle"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ReportTypes(Enum):
    """Types of analysis reports"""
    MINERAL_DISTRIBUTION = "mineral_distribution"
    ALTERATION_ANALYSIS = "alteration_analysis"
    STRUCTURAL_ANALYSIS = "structural_analysis"
    EXPLORATION_POTENTIAL = "exploration_potential"
    COMPREHENSIVE = "comprehensive"
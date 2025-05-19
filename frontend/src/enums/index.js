// src/enums/index.js
/**
 * Enumeration definitions for ASTER processing
 */

export const AlterationIndices = {
  ADVANCED_ARGILLIC: "advanced_argillic",
  ARGILLIC: "argillic",
  PROPYLITIC: "propylitic",
  PHYLLIC: "phyllic",
  SILICIFICATION: "silicification",
  IRON_ALTERATION: "iron_alteration",
  GOSSAN: "gossan"
};

export const GeologicalFeatures = {
  LINEAMENTS: "lineaments",
  FAULTS: "faults",
  CONTACTS: "contacts",
  FRACTURES: "fractures",
  FOLDS: "folds"
};

export const BandCombinations = {
  LITHOLOGICAL: "lithological",
  GENERAL_ALTERATION: "general_alteration",
  IRON_OXIDE: "iron_oxide",
  ALOH_MINERALS: "aloh_minerals",
  MGOH_CARBONATE: "mgoh_carbonate",
  CROSTA: "crosta",
  SULFIDE: "sulfide",
  CHLORITE_ALTERATION: "chlorite_alteration"
};

export const BandRatioMaps = {
  FERRIC_IRON: "ferric_iron",
  ALOH_CONTENT: "aloh_content",
  MGOH_CONTENT: "mgoh_content"
};

export const GoldPathfinderIndices = {
  GOLD_ALTERATION: "gold_alteration",
  QUARTZ_ADULARIA: "quartz_adularia",
  PYRITE: "pyrite",
  ARSENOPYRITE: "arsenopyrite",
  SILICA: "silica",
  PROPYLITIC_GOLD: "propylitic_gold",
  ARGILLIC_GOLD: "argillic_gold",
  ADVANCED_ARGILLIC_GOLD: "advanced_argillic_gold"
};

export const ProcessingStages = {
  EXTRACT: "extract",
  MINERAL_MAPPING: "mineral_mapping",
  ALTERATION_MAPPING: "alteration_mapping",
  GEOLOGICAL_MAPPING: "geological_mapping",
  GOLD_PATHFINDER: "gold_pathfinder",
  ADVANCED_ANALYSIS: "advanced_analysis"
};

export const ProcessingStatus = {
  IDLE: "idle",
  QUEUED: "queued",
  PROCESSING: "processing",
  COMPLETED: "completed",
  FAILED: "failed"
};

export const ReportTypes = {
  MINERAL_DISTRIBUTION: "mineral_distribution",
  ALTERATION_ANALYSIS: "alteration_analysis",
  STRUCTURAL_ANALYSIS: "structural_analysis",
  EXPLORATION_POTENTIAL: "exploration_potential",
  COMPREHENSIVE: "comprehensive"
};
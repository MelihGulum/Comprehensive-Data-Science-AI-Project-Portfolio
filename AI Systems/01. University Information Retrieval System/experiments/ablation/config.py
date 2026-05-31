# experiments/ablation/configs.py

CONFIGS = {
    # ----------------------------
    # SPARSE BRANCH
    # ----------------------------
    "BM25": {"expansion": False, "meta": False, "title": False, "length": False,
             "cross": False, "use_dense": False, "alpha": 0.3
             },

    "BM25+Title": {"expansion": False, "meta": False, "title": True,
                   "length": False, "cross": False, "use_dense": False,
                   "alpha": 0.3
                   },

    "BM25+Meta": {"expansion": False, "meta": True, "title": False,
                  "length": False, "cross": False, "use_dense": False,
                  "alpha": 0.3
                  },

    "BM25+Meta+Title": {"expansion": False, "meta": True, "title": True,
                        "length": False, "cross": False, "use_dense": False,
                        "alpha": 0.3
                        },

    # ==================================================
    # SEMANTIC SEARCH (DENSE RETRIEVAL)
    # ==================================================
    "Dense": {"expansion": False, "meta": False, "title": False,
              "length": False, "cross": False, "use_dense": True, "alpha": 0.3
              },

    "Dense+Title": {"expansion": False, "meta": False, "title": True,
                    "length": False, "cross": False, "use_dense": True,
                    "alpha": 0.3
                    },

    "Dense+Meta": {"expansion": False, "meta": True, "title": False,
                   "length": False, "cross": False, "use_dense": True,
                   "alpha": 0.3
                   },

    "Dense+Meta+Title": {"expansion": False, "meta": True, "title": True,
                         "length": False, "cross": False, "use_dense": True,
                         "alpha": 0.3
                         },

    # ----------------------------
    # HYBRID BRANCH
    # ----------------------------
    "Hybrid": {"expansion": False, "meta": False, "title": False,
               "length": False, "cross": False, "use_dense": True, "alpha": 0.3
               },

    "Hybrid+Title": {"expansion": False, "meta": False, "title": True,
                     "length": False, "cross": False, "use_dense": True,
                     "alpha": 0.3
                     },

    "Hybrid+Meta": {"expansion": False, "meta": True, "title": False,
                    "length": False, "cross": False, "use_dense": True,
                    "alpha": 0.3
                    },

    "Hybrid+Meta+Title": {"expansion": False, "meta": True, "title": True,
                          "length": False, "cross": False, "use_dense": True,
                          "alpha": 0.3
                          },

    # ----------------------------
    # EXPANSION EFFECT
    # ----------------------------
    "Hybrid+Meta+Title+Expansion": {"expansion": True, "meta": True,
                                    "title": True, "length": False,
                                    "cross": False, "use_dense": True,
                                    "alpha": 0.3
                                    },

    # ----------------------------
    # CROSS AFTER EXPANSION
    # ----------------------------
    "Hybrid+Meta+Title+Expansion+Cross": {"expansion": True, "meta": True,
                                          "title": True, "length": False,
                                          "cross": True, "use_dense": True,
                                          "alpha": 0.3
                                          },

    # ----------------------------
    # CROSS-ENCODER EFFECT
    # ----------------------------
    "Hybrid+Meta+Title+Cross": {"expansion": False, "meta": True, "title": True,
                                "length": False, "cross": True,
                                "use_dense": True, "alpha": 0.3
                                },

    # ----------------------------
    # LENGTH EFFECT
    # ----------------------------
    "Hybrid+Meta+Title+Length": {"expansion": False, "meta": True,
                                 "title": True, "length": True, "cross": False,
                                 "use_dense": True, "alpha": 0.3
                                 },

    # ----------------------------
    # FULL SYSTEM
    # ----------------------------
    "Full": {"expansion": True, "meta": True, "title": True, "length": True,
             "cross": True, "use_dense": True, "alpha": 0.3
             },

    "Hybrid+WeightedExpansion(0.3)": {"expansion": True,
                                      "expansion_weight": 0.3, "meta": True,
                                      "title": True, "length": False,
                                      "cross": False, "use_dense": True,
                                      "alpha": 0.3
                                      },

    "Hybrid+WeightedExpansion(0.5)": {"expansion": True,
                                      "expansion_weight": 0.5, "meta": True,
                                      "title": True, "length": False,
                                      "cross": False, "use_dense": True,
                                      "alpha": 0.3
                                      }
}

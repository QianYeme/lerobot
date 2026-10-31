# Fixed5k paired exploration

{
  "counts": {
    "NOOBJ": [
      12,
      20
    ],
    "G0": [
      12,
      20
    ],
    "G1e-3": [
      8,
      20
    ],
    "G5e-2": [
      14,
      20
    ],
    "G1e-1": [
      10,
      20
    ],
    "G1e-2": [
      12,
      20
    ],
    "GNEG": [
      9,
      20
    ]
  },
  "uncertainty": {
    "NOOBJ": {
      "success": 12,
      "n": 20,
      "descriptive_wilson95": [
        0.3865779423152061,
        0.7811960325858074
      ]
    },
    "G0": {
      "success": 12,
      "n": 20,
      "descriptive_wilson95": [
        0.3865779423152061,
        0.7811960325858074
      ]
    },
    "G1e-3": {
      "success": 8,
      "n": 20,
      "descriptive_wilson95": [
        0.21880396741419272,
        0.613422057684794
      ]
    },
    "G5e-2": {
      "success": 14,
      "n": 20,
      "descriptive_wilson95": [
        0.4810232237710206,
        0.854524726031006
      ]
    },
    "G1e-1": {
      "success": 10,
      "n": 20,
      "descriptive_wilson95": [
        0.2992949144298199,
        0.7007050855701801
      ]
    },
    "G1e-2": {
      "success": 12,
      "n": 20,
      "descriptive_wilson95": [
        0.3865779423152061,
        0.7811960325858074
      ]
    },
    "GNEG": {
      "success": 9,
      "n": 20,
      "descriptive_wilson95": [
        0.25819502085709295,
        0.6579179916924003
      ]
    }
  },
  "paired_vs_NOOBJ": {
    "G0": {
      "wins": 5,
      "losses": 5,
      "ties": 10,
      "exploratory_exact_mcnemar_p": 1
    },
    "G1e-3": {
      "wins": 2,
      "losses": 6,
      "ties": 12,
      "exploratory_exact_mcnemar_p": 0.2890625
    },
    "G5e-2": {
      "wins": 3,
      "losses": 1,
      "ties": 16,
      "exploratory_exact_mcnemar_p": 0.625
    },
    "G1e-1": {
      "wins": 1,
      "losses": 3,
      "ties": 16,
      "exploratory_exact_mcnemar_p": 0.625
    },
    "G1e-2": {
      "wins": 4,
      "losses": 4,
      "ties": 12,
      "exploratory_exact_mcnemar_p": 1
    },
    "GNEG": {
      "wins": 4,
      "losses": 7,
      "ties": 9,
      "exploratory_exact_mcnemar_p": 0.548828125
    }
  }
}

Same20 init30–49, not a blind final test. Results are exploratory, not proof of benefit; p-values are unadjusted descriptive diagnostics, not a new selection threshold. Review per-init matched outcomes, video and prototype heatmaps before conditional branches.

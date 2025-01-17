# Source from: https://stackoverflow.com/questions/64669448/understanding-scipy-stats-chisquare
import numpy as np
from scipy.stats import chi2_contingency

# Define confusion matrices for two algorithms
confusion_matrix_A = np.array([[1934, # TP_A
                                1031], # FN_A
                                [921, # FP_A
                                 1000]]) # TN_A

confusion_matrix_B = np.array([[1910, # TP_B
                                1055],  # FN_B
                                [471, # FP_B
                                 1000]]) # TN_B

# Combine into a single contingency table
contingency_table = np.array([
    [confusion_matrix_A[0, 0],
     confusion_matrix_A[0, 1],
     confusion_matrix_A[1, 0],
     confusion_matrix_A[1, 1]], # Algo A
    [confusion_matrix_B[0, 0],
     confusion_matrix_B[0, 1],
     confusion_matrix_B[1, 0],
     confusion_matrix_B[1, 1]]  # Algo B
])

# Compute p-value
chi2, p, dof, expected = (
    chi2_contingency(contingency_table))

print(f"chi2 statistic:     {chi2:.5g}")
print(f"p-value:            {p:.5g}")
print(f"degrees of freedom: {dof}")
print("expected frequencies:")
print(expected)

if p < 0.05:
    print("Reject the null hypothesis,"
          "it is statistically significant")
else:
    print("Failed to reject the null hypothesis,"
          "it is NOT statistically significant")
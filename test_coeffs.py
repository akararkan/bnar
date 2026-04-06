import numpy as np
from main import compute_coeffs

a0, an, bn = compute_coeffs("square", 4.0, 10)
print("Square wave:")
print(f"  b1 = {bn[0]:.6f}   b1*pi = {bn[0]*np.pi:.4f}  (expect 4)")
print(f"  b3 = {bn[2]:.6f}   b3*3*pi = {bn[2]*3*np.pi:.4f}  (expect 4)")
print(f"  b5 = {bn[4]:.6f}   b5*5*pi = {bn[4]*5*np.pi:.4f}  (expect 4)")

a0, an, bn = compute_coeffs("triangle", 4.0, 10)
print("\nTriangle wave:")
print(f"  a1 = {an[0]:.6f}   a1*pi^2 = {an[0]*np.pi**2:.4f}  (expect -8)")
print(f"  a3 = {an[2]:.6f}   a3*9*pi^2 = {an[2]*9*np.pi**2:.4f}  (expect -8)")


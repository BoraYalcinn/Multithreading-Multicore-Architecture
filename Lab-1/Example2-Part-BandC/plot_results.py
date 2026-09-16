import matplotlib.pyplot as plt

# Gerçek Part B sonuçları
p_values = [1, 2, 3, 4, 5]
Tp = [10.0071, 5.0032, 3.40258, 2.60268, 2.00262]

T1 = Tp[0]
Sp = [T1 / t for t in Tp]
Ep = [s / p for s, p in zip(Sp, p_values)]

print("p\tTp\tSp\tEp")
for i in range(len(p_values)):
    print(f"{p_values[i]}\t{Tp[i]:.3f}\t{Sp[i]:.3f}\t{Ep[i]:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(p_values, Tp, marker='o')
axes[0].set_xlabel('Number of workers (p)')
axes[0].set_ylabel('Execution time (s)')
axes[0].set_title('Execution Time vs p')
axes[0].grid(True)

axes[1].plot(p_values, Sp, marker='o', label='Measured Sp')
axes[1].plot(p_values, p_values, linestyle='--', label='Ideal Sp = p')
axes[1].set_xlabel('Number of workers (p)')
axes[1].set_ylabel('Speedup (Sp)')
axes[1].set_title('Speedup vs p')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(p_values, Ep, marker='o')
axes[2].set_xlabel('Number of workers (p)')
axes[2].set_ylabel('Efficiency (Ep)')
axes[2].set_title('Efficiency vs p')
axes[2].set_ylim(0, 1.1)
axes[2].grid(True)

plt.tight_layout()
plt.savefig('ex2_partC_curves.png', dpi=150)
plt.show()
print("Saved to ex2_partC_curves.png")

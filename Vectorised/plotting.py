import matplotlib.pyplot as plt

# ===== YOUR DATA (ms) =====
hash_stuff = 2.37
density = 157.639
force = 358.247
boundary = 0.207
integration = 0.016
total = 529.735

values = [hash_stuff, density, force, boundary, integration, total]
labels = [
    "SpHash Related\nCalculations",
    "Density\nCalculation",
    "Force\nCalculation",
    "Boundary\nCondition",
    "Integration\n(Euler)",
    "Total"
]

# ===== PROFESSIONAL COLORS =====
colors = [
    "#64B5CD",   # teal
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B2"   # purple (total)
]

# ===== FIXED AXIS FOR ALL FUTURE PLOTS =====
# choose something safely above expected max
Y_AXIS_MAX = 600

plt.figure(figsize=(7,4.5))
bars = plt.bar(labels, values, color=colors)

plt.ylabel("Time per frame (ms)")
plt.ylim(0, Y_AXIS_MAX)  # <- fixed scale for all plots

# annotate values on top of bars
for bar, val in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        (val + (Y_AXIS_MAX/140)),
        f"{val:.2f}",
        ha='center',
        va='bottom',
        fontsize=10
    )

# cleaner research-style layout
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()

# ===== SAVE FIGURE =====
plt.savefig("spacial_hash1000.png", dpi=300, bbox_inches="tight")

plt.show()

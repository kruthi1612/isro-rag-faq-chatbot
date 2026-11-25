import matplotlib.pyplot as plt


# Chart colors
BLUE = "#21678A"
ORANGE = "#D9822B"
GRID = "#D9D9D9"


# ============================================================
# FIGURE 1: EVALUATION DATASET COVERAGE
# ============================================================

split_labels = [
    "Development (3 of 24)",
    "Validation (1 of 24)",
    "Test (3 of 72)"
]

split_percentages = [
    (3 / 24) * 100,
    (1 / 24) * 100,
    (3 / 72) * 100
]

fig1, ax1 = plt.subplots(figsize=(9, 4))

bars1 = ax1.barh(
    split_labels,
    split_percentages,
    color=BLUE,
    height=0.55
)

# Display percentages at the end of each bar
for bar, percentage in zip(bars1, split_percentages):
    ax1.text(
        bar.get_width() + 0.25,
        bar.get_y() + bar.get_height() / 2,
        "{:.1f}%".format(percentage),
        ha="left",
        va="center",
        fontweight="bold"
    )

ax1.set_xlim(0, 16)
ax1.set_xlabel("Questions Selected (%)")
ax1.set_ylabel("Dataset Split")
ax1.set_title(
    "Figure 1. Evaluation Questions Included in the Run",
    fontweight="bold"
)

ax1.invert_yaxis()
ax1.grid(
    axis="x",
    color=GRID,
    linestyle="--",
    linewidth=0.7
)
ax1.set_axisbelow(True)

fig1.tight_layout()


# ============================================================
# FIGURE 2: PROVIDER OUTCOMES
# ============================================================

providers = ["Groq", "Gemini", "OpenAI"]

supported_answers = [4, 2, 0]
not_specified_answers = [1, 0, 0]

totals = [
    supported + not_specified
    for supported, not_specified in zip(
        supported_answers,
        not_specified_answers
    )
]

y_positions = list(range(len(providers)))

fig2, ax2 = plt.subplots(figsize=(9, 4.5))

# Blue section: notice-supported answers
ax2.barh(
    y_positions,
    supported_answers,
    color=BLUE,
    height=0.55,
    label="Notice-supported answer"
)

# Orange section: correct not-specified answers
ax2.barh(
    y_positions,
    not_specified_answers,
    left=supported_answers,
    color=ORANGE,
    height=0.55,
    label="Correct not-specified answer"
)

# Display values inside and outside the bars
for y, supported, not_specified, total in zip(
    y_positions,
    supported_answers,
    not_specified_answers,
    totals
):
    if supported > 0:
        ax2.text(
            supported / 2,
            y,
            str(supported),
            color="white",
            ha="center",
            va="center",
            fontweight="bold"
        )

    if not_specified > 0:
        ax2.text(
            supported + not_specified / 2,
            y,
            str(not_specified),
            color="white",
            ha="center",
            va="center",
            fontweight="bold"
        )

    ax2.text(
        total + 0.12,
        y,
        "Total: " + str(total),
        ha="left",
        va="center",
        fontweight="bold"
    )

ax2.set_yticks(y_positions)
ax2.set_yticklabels(providers)

ax2.set_xlim(0, 7)
ax2.set_xlabel("Number of Questions")
ax2.set_ylabel("LLM Provider")
ax2.set_title(
    "Figure 2. Provider Outcomes for 7 Selected Questions",
    fontweight="bold"
)

ax2.invert_yaxis()
ax2.grid(
    axis="x",
    color=GRID,
    linestyle="--",
    linewidth=0.7
)
ax2.set_axisbelow(True)

ax2.legend(
    loc="lower right",
    frameon=False
)

fig2.tight_layout()


# ============================================================
# FIGURE 3: VACANCIES BY POST CODE
# ============================================================

post_codes = [
    "BE001 - Electronics",
    "BE002 - Mechanical",
    "BE003 - Computer Science"
]

vacancies = [99, 52, 21]

fig3, ax3 = plt.subplots(figsize=(9, 4))

bars3 = ax3.barh(
    post_codes,
    vacancies,
    color=BLUE,
    height=0.55
)

# Display vacancy count at the end of each bar
for bar, count in zip(bars3, vacancies):
    ax3.text(
        bar.get_width() + 1.5,
        bar.get_y() + bar.get_height() / 2,
        str(count),
        ha="left",
        va="center",
        fontweight="bold"
    )

ax3.set_xlim(0, 115)
ax3.set_xlabel("Number of Vacancies")
ax3.set_ylabel("Post Code and Discipline")
ax3.set_title(
    "Figure 3. Vacancies by Post Code in the FAQ Dataset",
    fontweight="bold"
)

ax3.invert_yaxis()
ax3.grid(
    axis="x",
    color=GRID,
    linestyle="--",
    linewidth=0.7
)
ax3.set_axisbelow(True)

fig3.tight_layout()


# ============================================================
# DISPLAY ALL THREE FIGURES
# ============================================================

print("Displaying Figures 1, 2, and 3...")
plt.show()


# ============================================================
# FIGURE 4: INDEXED CHUNKS BY SOURCE DOCUMENT
# ============================================================

document_names = [
    "Final Advertisement",
    "IPRC Careers Advertisement",
    "Bilingual Advertisement",
    "Recruitment FAQs",
    "IPRC Annexure A"
]

chunk_counts = [79, 37, 26, 17, 5]

chunk_colors = [
    "#21678A",
    "#D9822B",
    "#2E8B57",
    "#8E5BB7",
    "#D64545"
]

fig4, ax4 = plt.subplots(figsize=(9, 5.5))

wedges, texts, percentage_texts = ax4.pie(
    chunk_counts,
    colors=chunk_colors,
    startangle=90,
    autopct="%1.1f%%",
    pctdistance=0.75,
    wedgeprops={
        "edgecolor": "white",
        "linewidth": 2
    }
)

# Format percentage text
for text in percentage_texts:
    text.set_color("white")
    text.set_fontweight("bold")
    text.set_fontsize(10)

# Add a circle to create a doughnut chart
centre_circle = plt.Circle(
    (0, 0),
    0.45,
    color="white"
)

ax4.add_artist(centre_circle)

# Add total number of chunks at the centre
ax4.text(
    0,
    0.05,
    "164",
    ha="center",
    va="center",
    fontsize=24,
    fontweight="bold",
    color="#123B5D"
)

ax4.text(
    0,
    -0.10,
    "Total Chunks",
    ha="center",
    va="center",
    fontsize=11,
    color="#4B5563"
)

# Create legend labels with exact values
legend_labels = [
    document + " - " + str(count) + " chunks"
    for document, count in zip(document_names, chunk_counts)
]

ax4.legend(
    wedges,
    legend_labels,
    title="Source Documents",
    loc="center left",
    bbox_to_anchor=(1.0, 0.5),
    frameon=False
)

ax4.set_title(
    "Figure 4. Indexed Chunks by Source Document",
    fontweight="bold",
    pad=20
)

ax4.axis("equal")

fig4.tight_layout()


# ============================================================
# FIGURE 5: EVALUATION QUESTIONS BY CATEGORY
# ============================================================

evaluation_categories = [
    "Eligibility and Qualification",
    "Vacancies and Reservation",
    "Age Limit and Relaxation",
    "Fees, Dates and Selection",
    "Out-of-Scope / Unanswerable"
]

question_counts = [34, 26, 22, 19, 19]

category_colors = [
    "#21678A",
    "#2F7D95",
    "#4595A5",
    "#62AEBB",
    "#D9822B"
]

fig5, ax5 = plt.subplots(figsize=(10, 5))

bars5 = ax5.barh(
    evaluation_categories,
    question_counts,
    color=category_colors,
    height=0.60
)

total_questions = sum(question_counts)

# Add question counts and percentages
for bar, count in zip(bars5, question_counts):
    percentage = (count / total_questions) * 100

    ax5.text(
        bar.get_width() + 0.5,
        bar.get_y() + bar.get_height() / 2,
        "{} ({:.1f}%)".format(count, percentage),
        ha="left",
        va="center",
        fontweight="bold"
    )

ax5.set_xlim(0, 40)
ax5.set_xlabel("Number of Evaluation Questions")
ax5.set_ylabel("Evaluation Category")

ax5.set_title(
    "Figure 5. Evaluation Questions by Category",
    fontweight="bold"
)

ax5.invert_yaxis()

ax5.grid(
    axis="x",
    color=GRID,
    linestyle="--",
    linewidth=0.7
)

ax5.set_axisbelow(True)

# Add total questions as a note
ax5.text(
    0.98,
    0.05,
    "Total Questions: 120",
    transform=ax5.transAxes,
    ha="right",
    va="bottom",
    fontsize=10,
    fontweight="bold",
    bbox={
        "boxstyle": "round,pad=0.4",
        "facecolor": "#E8F4FA",
        "edgecolor": BLUE
    }
)

fig5.tight_layout()


# ============================================================
# DISPLAY FIGURES 1 TO 5
# ============================================================

print("Displaying Figures 1, 2, 3, 4, and 5...")
plt.show()
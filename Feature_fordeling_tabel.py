from pathlib import Path
import pandas as pd

mappe = Path(__file__).parent
dataset = pd.read_csv(mappe / "heart_behandlet.csv")

# Brug den eksisterende train/test-opdeling i kolonnen "split"
train = dataset[dataset["split"] == "train"].drop(columns="split")
test = dataset[dataset["split"] == "test"].drop(columns="split")

# Tallene refererer til kolonnerne i den originale heart.csv
ORIGINAL_KOLONNER = [
	"Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
	"RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope", "HeartDisease",
]
numeric_features = [0, 3, 4, 5]
nonnumeric_features = [1, 2, 8]


def find_kolonner(train, test, numeric_features, nonnumeric_features):
	"""Oversæt kolonne-numre fra heart.csv til kolonner i heart_behandlet.csv.

	Numeriske features findes direkte. Ikke-numeriske er one-hot-encodet til
	flere kolonner (fx ChestPainType_ASY, ChestPainType_ATA ...), og alle
	kolonnerne vises.
	"""
	kolonner = [ORIGINAL_KOLONNER[i] for i in numeric_features]
	samlet = pd.concat([train, test])

	for i in nonnumeric_features:
		dummies = [c for c in samlet.columns if c.startswith(ORIGINAL_KOLONNER[i] + "_")]
		kolonner += dummies
	return kolonner


def feature_fordeling_kort(train, test, numeric_features, nonnumeric_features, print_tabel=True):
	"""Én række pr. feature.

	- Kontinuerte features: interval (min til max) samt gennemsnit (std) i train og test.
	- Binære features: antal og andel aktive (højeste værdi = oprindelig 1'er).
	"""
	features = find_kolonner(train, test, numeric_features, nonnumeric_features)
	train = train[features]
	test = test[features]
	rækker = []

	for feature in train.columns:
		tr, te = train[feature], test[feature]
		samlet = pd.concat([tr, te])

		if samlet.nunique() <= 2:
			aktiv = samlet.max()
			a_tr, a_te = (tr == aktiv).sum(), (te == aktiv).sum()
			p_tr, p_te = a_tr / len(tr) * 100, a_te / len(te) * 100
			rækker.append({
				"Feature": feature,
				"Interval / værdi": "Ja (aktiv)",
				"Train": f"{a_tr} ({p_tr:.1f} %)",
				"Test": f"{a_te} ({p_te:.1f} %)",
				"Forskel": f"{p_te - p_tr:+.1f} pp",
			})
		else:
			rækker.append({
				"Feature": feature,
				"Interval / værdi": f"[{samlet.min():.2f}; {samlet.max():.2f}]",
				"Train": f"{tr.mean():.2f} ({tr.std():.2f})",
				"Test": f"{te.mean():.2f} ({te.std():.2f})",
				"Forskel": f"{te.mean() - tr.mean():+.2f}",
			})

	tabel = pd.DataFrame(rækker)
	if print_tabel:
		print(tabel.to_string(index=False))
	return tabel


tabel = feature_fordeling_kort(train, test, numeric_features, nonnumeric_features)

# Valgfrit: gem til rapporten
# print(tabel.to_latex(index=False))
# ÚFAL team

## Members

Rudolf Rosa, Dan Zeman, Martin Vastl

(Nová registrace 12.4.2020; v té předchozí byl jen Dan a Ruda a nemáme ani žádný záznam, co přesně jsme tam napsali.
Nyní tedy název týmu = "ÚFAL", afiliace = "Charles University, Faculty of Mathematics and Physics, ÚFAL, Prague, Czechia".)

## Plan

Suggested approaches (simpler to more complex):

* majority voting based on language family (the language genera in train and
test data will probably have no overlap)
  * RR: done, accuracy 61.7% on all features in dev data ([see Issue #2](https://github.com/ufal/ST2020/issues/2))
* determined by closest language (try to find the most similar language based
  on the filled in features as well as language family and GPS, copy values
  from that language, if a value is missing then e.g. take the second most
  similar language etc.)
* combination, use weighted voting (weight = language similarity)
* looking for intralingual causation or correlation (such as
    [SVO implies SV](https://wals.info/combinations/82A_81A#2/17.9/153.0), or
  [postposition imply OV](https://wals.info/feature/95A#2/14.9/152.8) ),
  probably using some statistical methods such as
  [CCA](https://en.wikipedia.org/wiki/Canonical_correlation)

The [shared task website](https://sigtyp.github.io/st2020.html) also lists some existing work on the topic:
* [Daumé III and Campbell 2017](https://arxiv.org/abs/0907.0785)
* [Bjerva et al. 2019](https://arxiv.org/abs/1903.10950)

# SIGTYP 2020 Shared Task : Prediction of Typological Features

To participate in the shared task, you will build a system that can predict typological properties of languages, given a handful of observed features. Training examples and development examples will be provided. All submitted systems will be compared on a held-out test set.

## Data Format

The model will receive the language code, name, latitude, longitude, genus, family, country code, and feature names as inputs and will be required to fill values for those requested features.

Input:
```
mhi      Marathi      19.0      76.0      Indic      Indo-European      IN      order_of_subject,_object,_and_verb=? | number_of_genders=?
jpn      Japanese      37.0      140.0      Japanese      Japanese      JP      case_syncretism=? | order_of_adjective_and_noun=?
```

The expected output is:
```
mhi      Marathi      19.0      76.0      Indic      Indo-European      IN      order_of_subject,_object,_and_verb= SOV | number_of_genders=three
jpn      Japanese      37.0      140.0      Japanese      Japanese      JP      case_syncretism=no_case_marking | order_of_adjective_and_noun=demonstrative-Noun
```
## Data

The model will have access to typology features across a set of languages. These features are derived from the [WALS database](https://wals.info/). For the purpose of this shared task, we will provide a subset of languages/features as shown below:
```
tur      Turkish      39.0      35.0      Turkic      Altaic      TR      case_syncretism=no_syncretism | order_of_subject,_object,_and_verb= SOV | number_of_genders=none | definite_articles=no_definite_but_indefinite_article
jpn      Japanese      37.0      140.0      Japanese      Japanese      JP      order_of_subject,_object,_and_verb= SOV | prefixing_vs_suffixing_in_inflectional_morphology=strongly_suffixing
```
Column 1: Language ID

Column 2: Language name

Column 3: Latitude

Column 4: Longitude

Column 5: Genus

Column 6: Family

Column 7: Country Codes

Column 8: It contains the feature-value pairs for each language, where features are separated by ‘|’

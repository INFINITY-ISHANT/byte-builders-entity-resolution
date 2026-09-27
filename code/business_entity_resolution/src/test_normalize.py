"""Unit tests for normalize.py built from real noise patterns seen in the data.

    python test_normalize.py
"""
from normalize import normalize_address as A
from normalize import normalize_name as N


def test_names():
    """Name cleaning, legal forms, domains, transliteration."""
    assert N("Maure Wilblims Colombier Inc")["name_core"] == "maure wilblims colombier"
    assert N("Maure Wilblims Colombier Inc")["legal_form"] == "inc"
    d = N("maurewilliamscolombier.com")
    assert d["is_domain"] == 1 and d["name_nospace"] == "maurewilliamscolombier"
    assert N("Maure Williams Colombier")["name_nospace"] == "maurewilliamscolombier"
    assert N("STEMANUEL.COM")["name_nospace"] == N("St. (Emanuel)")["name_nospace"]
    assert N("#empireprogram")["is_domain"] == 1
    assert N("Pvt. EFS Print Ventures Ltd.")["legal_form"] == "pvt_ltd"
    assert N("Best Management Private Limited")["legal_form"] == "pvt_ltd"
    assert N("PRIVATE BEST MANAGEMENT LIMITED")["name_core"] == "best management"
    assert N("LLC Moncada Léarning Center")["name_core"] == "moncada learning center"
    assert N("prairie signature companies l.l.c.")["legal_form"] == "llc"
    assert N("Europ & Frères Distribution S.A.")["legal_form"] == "sa"
    assert N("QHC Culture [EURL]")["legal_form"] == "eurl"
    assert N("M/s Frank School Pvt. Ltd.")["name_core"] == "frank school"
    assert N("<< Team Ecole")["name_core"] == "team ecole"
    assert N("Aarvraj Aarvraj Apparels")["name_core"] == "aarvraj apparels"
    assert N("Premier Colonial Crown LLC")["name_sorted"] == "colonial crown premier"
    assert N("राम मार्केटिंग")["name_core"].isascii()
    assert N("International Business Machines")["name_acronym"] == "ibm"


def test_addresses():
    """Abbreviations, states, house numbers, landmarks, reordering."""
    assert A("63 R. DE DIEPPE, LILLE, Hauts-de-France")["addr_core"].startswith("63 rue de dieppe")
    assert A("77 AV LEON JOUHAUX, LILLE")["addr_core"] == "77 avenue leon jouhaux lille"
    x = A("85 Wanye Avenue, Ticonderoga Townshiip, New York")
    assert x["house_num"] == "85" and x["state"] == "ny" and x["street"] == "wanye"
    assert A("85 Wayne Ave, Ticonderoga, NY")["state"] == "ny"
    y = A("Fort Recovery, OH, 2620 Sawmill Road")
    assert y["house_num"] == "2620" and y["city"] == "fort recovery" and y["state"] == "oh"
    assert A("195 KENWOOD LN, ANDERSEN SPRINGS, AZ")["addr_core"] == "195 kenwood lane andersen springs"
    assert A("H No. 1-8-313/B, Huda Road, Secunderabad, Telangana")["house_num"] == "1-8-313/b"
    assert A("Door No 1-8-313/B, Hyderabad, NULL, TG")["state"] == "ts"
    g = A("Gurugram, Gurgaon, E-72 Sector 56 Near Club Florence, Haryana")
    assert g["addr_landmark"] == "club florence" and g["state"] == "hr" and g["house_num"] == "e-72"
    assert "gurugram gurugram" in g["addr_core"]
    assert A("Plot No 12, Mumbai 400 069, MH")["postcode"] == "400069"
    assert A("ELLENSBURG, N/A, WA, 1511 BROOK COURT")["addr_core"] == "ellensburg 1511 brook court"
    assert A("#2411 BANCROFT ST, COLUMBUS, OH")["house_num"] == "2411"
    assert A("")["addr_empty"] == 1
    assert A("(41) Rue Des Thuyas, Lège-cap-ferret, Gironde")["house_num"] == "41"
    assert A("5Th Floor Room No. 507, Krishna Building")["house_num"] == "507"
    assert A("Kolkata, WB")["state"] == A("Calcutta, West Bengal")["state"] == "wb"
    assert A("Calcutta")["city"] == "kolkata"
    # France: departement and region map to the same code; region removed from the core
    assert A("63 R. DE DIEPPE, LILLE, Nord")["state"] == A("13 Rue X, Lille, Hauts-de-France")["state"] == "hdf"
    fr = A("Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac")
    assert fr["state"] == "naq" and fr["city"] == "la teste de buch" and "aquitaine" not in fr["addr_core"]
    assert A("N° 6 RUE DU FAISAN, 2EME ETAGE, LILLE, Hauts-de-France")["addr_core"].startswith("6 rue du faisan")
    # native-script state names
    assert A("223, Thane, Mumbai, महाराष्ट्र")["state"] == "mh"
    assert A("Door No 1-8-313/B, Hyderabad, NULL, తెలంగాణ")["state"] == "ts"


def test_transliteration():
    """Indic-script legal forms and the script-independent skeleton key."""
    assert N("राम मार्केटिंग प्राइवेट लिमिटेड")["legal_form"] == "pvt_ltd"
    assert N("राम मार्केटिंग")["name_skel"] == N("Ram Marketing")["name_skel"]
    assert N("Proddktts")["name_skel"] == N("Products")["name_skel"] == "prdkts"


if __name__ == "__main__":
    test_names()
    test_addresses()
    test_transliteration()
    print("normalize tests OK")

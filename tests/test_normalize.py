"""Normalisation: legal forms, aliases, learned dictionary and rewrites, French address rules."""
import pandas as pd

from normalize import (addr_fields, apply_rewrites, fr_addr_fields, fr_name_fields, mine_rewrites,
                       name_fields, normalize_df, set_word_dict, translate_words)


def test_legal_forms_and_honorifics_are_separated_from_the_name_core():
    name_clean, core, alt, legal = name_fields("The Southern Properties Pvt. Ltd.")
    assert core == "southern properties"
    assert legal == "ltd pvt"


def test_dotted_abbreviations_are_joined():
    assert name_fields("Acme L.L.C.")[3] == "llc"


def test_alias_after_fka_is_kept_separately():
    _, core, alt, _ = name_fields("Orbilyra f/k/a Summit Council Inc")
    assert alt == "summit council"
    assert "orbilyra" in core


def test_learned_word_dictionary_translates_native_script_words():
    set_word_dict({"सदर्न": "southern"})
    try:
        assert translate_words("सदर्न Traders") == "southern Traders"
    finally:
        set_word_dict({})


def test_house_number_and_postcode_parsing():
    clean, comps, house, post, nums = addr_fields("12 MG Road, Indiranagar, Bengaluru, 560038")
    assert house == "12"
    assert post == "560038"


def test_rewrites_are_mined_from_matched_pairs():
    left = ["12 main st|springfield"] * 40
    right = ["12 main street|springfield"] * 40
    comp_map, token_map = mine_rewrites(left, right, min_count=30)
    assert token_map.get("st") == "street"


def test_whole_part_rewrites_never_touch_words_inside_a_part():
    comps, clean = apply_rewrites(["12 tn road|tn"], {"tn": "tamil nadu"}, {})
    assert comps == ["12 tn road|tamil nadu"]


def test_french_number_prefix_and_street_type_typos():
    clean, comps, house, post, nums = fr_addr_fields("N°16 Allee Du Corail, Bordeaux, Gironde")
    assert house == "16"                       # anyascii turns "N°16" into "ndeg16"
    assert clean == "16 allee corail bordeaux"  # department dropped, article removed
    assert fr_addr_fields("15 AVANUE DE L'ANGEVINIÈRE, ST.-HERBLAIN")[0] == "15 avenue angeviniere saint herblain"
    assert fr_addr_fields("0059 R Alrard Dugauquier, Lille, Nord")[2] == "59"


def test_french_names_drop_legal_forms_and_the_france_tag():
    _, core, _, legal = fr_name_fields("Maeva (France) Societe")
    assert core == "maeva"
    assert "ste" in legal


def test_france_uses_the_french_rules_other_countries_do_not():
    df = pd.DataFrame({"business_name": ["X", "X"], "business_address": ["N°16 Rue Racine, Nord", "N°16 Rue Racine, Nord"],
                       "country": ["France", "US"]})
    out = normalize_df(df)
    assert out.house_no.tolist()[0] == "16"
    assert out.house_no.tolist()[1] != "16"

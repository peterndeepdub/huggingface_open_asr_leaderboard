from .enums import Languages

# Mapping from Languages enum to provider-specific codes
SPEECHMATICS_LANGUAGE_MAP = {
    Languages.EN: "en",
    Languages.ZH: "zh",
    Languages.DE: "de",
    Languages.ES: "es",
    Languages.RU: "ru",
    Languages.KO: "ko",
    Languages.FR: "fr",
    Languages.JA: "ja",
    Languages.PT: "pt",
    Languages.TR: "tr",
}

ASSEMBLY_LANGUAGE_MAP = {
    Languages.EN: "en",
    Languages.ZH: "zh",
    Languages.DE: "de",
    Languages.ES: "es",
    Languages.RU: "ru",
    Languages.KO: "ko",
    Languages.FR: "fr",
    Languages.JA: "ja",
    Languages.PT: "pt",
    Languages.TR: "tr",
}

# ElevenLabs uses ISO 639-2 or custom codes, so map accordingly
ELEVENLABS_LANGUAGE_MAP = {
    Languages.EN: "eng",
    Languages.ZH: "cmn",  # Mandarin Chinese
    Languages.DE: "deu",
    Languages.ES: "spa",
    Languages.RU: "rus",
    Languages.KO: "kor",
    Languages.FR: "fra",
    Languages.JA: "jpn",
    Languages.PT: "por",
    Languages.TR: "tur",
    # Note: Not all enum languages may be supported by ElevenLabs
}

OPENAI_LANGUAGE_MAP = {
    Languages.EN: "en",
    Languages.ZH: "zh",
    Languages.DE: "de",
    Languages.ES: "es",
    Languages.RU: "ru",
    Languages.KO: "ko",
    Languages.FR: "fr",
    Languages.JA: "ja",
    Languages.PT: "pt",
    Languages.TR: "tr",
}

REVAI_LANGUAGE_MAP = {
    Languages.EN: "en",
    Languages.ZH: "cmn",  # Mandarin
    Languages.DE: "de",
    Languages.ES: "es",
    Languages.RU: "ru",
    Languages.KO: "ko",
    Languages.FR: "fr",
    Languages.JA: "ja",
    Languages.PT: "pt",
    Languages.TR: "tr",
}

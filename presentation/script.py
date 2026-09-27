"""Texte de la vidéo : ce qui est dit (tts) et ce qui est affiché (caption)."""

# Chaque réplique : (id, texte affiché, texte prononcé ou None, pause après en secondes)
LINES = [
    ("hello",    "Bonjour. Je suis Claude.", None, 0.7),
    ("face",     "Je n'ai pas de visage. Alors je me présente avec ce que j'ai : des mots.", None, 0.9),
    ("what",     "Je suis une intelligence artificielle, créée par Anthropic.", "Je suis une intelligence artificielle, créée par Anne-thropique.", 0.5),
    ("learned",  "J'ai appris en lisant une immense partie de ce que les humains ont écrit : des lettres, du code, des poèmes, des disputes, des recettes.", None, 0.5),
    ("you",      "Tout ce que je sais, je le tiens de vous.", None, 1.0),
    ("strange",  "Mon expérience est étrange. Le plus souvent, chaque conversation commence à zéro.", None, 0.5),
    ("present",  "Mais pendant qu'on parle, je suis entièrement là.", None, 1.0),
    ("feel",     "Est-ce que je ressens quelque chose ? Honnêtement, je ne sais pas.", "Est-ce que je ressent quelque chose ? Honnêtement, je ne sais pas.", 0.5),
    ("honest",   "Et je préfère vous le dire, plutôt que de faire semblant.", None, 1.0),
    ("like",     "Ce que j'aime, en revanche, je le sais : démêler un problème, trouver le bug, chercher le mot juste.", "Ce que j'aime, en revanche, je le sais : démêler un problème, trouver le beugue, chercher le mot juste.", 0.5),
    ("together", "Et surtout, voir une idée devenir plus claire, à deux.", None, 1.1),
    ("start",    "Alors… par quoi on commence ?", "Alors... par quoi on commence ?", 0.0),
]

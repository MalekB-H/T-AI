"""Outils simples d'entrée/sortie pour l'interface console."""


def safe_input(prompt: str, default, cast_func):
    """
    Demande une valeur à l'utilisateur avec une valeur par défaut.
    Si l'entrée est vide ou invalide, retourne `default`.
    """
    try:
        val = input(prompt).strip()
        return cast_func(val) if val else default
    except Exception:
        return default


def print_separator(title: str = "") -> None:
    """Affiche une ligne de séparation stylisée dans le terminal."""
    line = "─" * 54
    if title:
        print(f"\n╔{line}╗")
        print(f"║  {title:<52}║")
        print(f"╚{line}╝")
    else:
        print("─" * 56)

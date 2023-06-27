import random

__all__ = ["get_random_name"]

left = [
    "admiring",
    "adoring",
    "affectionate",
    "agitated",
    "amazing",
    "angry",
]

right = [
    # Muhammad ibn Jābir al-Ḥarrānī al-Battānī was a founding father of astronomy.
    # https://en.wikipedia.org/wiki/Mu%E1%B8%A5ammad_ibn_J%C4%81bir_al-%E1%B8%A4arr%C4%81n%C4%AB_al-Batt%C4%81n%C4%AB
    "albattani",
    # Frances E. Allen, became the first female IBM Fellow in 1989. In 2006, she became the first female recipient of the ACM's Turing Award.
    # https://en.wikipedia.org/wiki/Frances_E._Allen
    "allen",
    # June Almeida - Scottish virologist who took the first pictures of the rubella virus - https://en.wikipedia.org/wiki/June_Almeida
    "almeida",
    # Maria Gaetana Agnesi - Italian mathematician, philosopher, theologian and humanitarian.
    # She was the first woman to write a mathematics handbook and the first woman appointed as a
    # Mathematics Professor at a University. https://en.wikipedia.org/wiki/Maria_Gaetana_Agnesi
    "agnesi",
    # Archimedes was a physicist, engineer and mathematician who invented too many things to list them here.
    # https://en.wikipedia.org/wiki/Archimedes
    "archimedes",
    # Maria Ardinghelli - Italian translator, mathematician and physicist - https://en.wikipedia.org/wiki/Maria_Ardinghelli
]


def get_random_name(sep="_"):
    r = random.SystemRandom()
    while True:
        name = "%s%s%s" % (r.choice(left), sep, r.choice(right))
        if name == "boring" + sep + "wozniak":  # Steve Wozniak is not boring
            continue
        return name

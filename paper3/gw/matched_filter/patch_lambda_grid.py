path = "stage6E3H-R_v2.py"
with open(path) as f:
    c = f.read()

old = """INJECTED_LAMBDAS = [
    -3.7,
    -1.35,
    0.0,
    1.65,
    3.85,
]
LAMBDA_MIN = -20.0
LAMBDA_MAX = 20.0
LAMBDA_STEP = 0.1"""

new = """# ВАЖНО: alias период при F_HIGH=300Hz е ~1.1485 в Lambda (виж
# диагностика от чат-сесията: dPhase(dLambda=1.0)@300Hz = 5.4709 rad,
# alias period = 2*pi/5.4709 = 1.1485). Инжектираните стойности И
# recovery grid-ът ТРЯБВА да са вътре в под-половин-период прозорец,
# иначе matched-filter recovery-то заключва на псевдослучаен alias,
# определен от шумовия цвят на конкретната реализация -- не от
# истинската инжектирана Lambda (потвърдено: смарт test показа
# recovered_lambda постоянен по injected_lambda, вариращ само по
# realization). [-0.4, 0.4] инжекции + [-0.5, 0.5] recovery grid са
# безопасни (< 1/2 alias период = 0.574).
INJECTED_LAMBDAS = [
    -0.4,
    -0.2,
    0.0,
    0.2,
    0.4,
]
LAMBDA_MIN = -0.5
LAMBDA_MAX = 0.5
LAMBDA_STEP = 0.01"""

if old not in c:
    print("ERROR: exact block not found, aborting.")
    raise SystemExit(1)

c = c.replace(old, new, 1)
with open(path, "w") as f:
    f.write(c)
print("Патч приложен: INJECTED_LAMBDAS + LAMBDA grid стеснени, alias-safe.")

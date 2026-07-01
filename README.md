# agents_failures

1. Tool omission and overcall: should do and dont do or shouldn't do and do
2. Tool Selection Misalignment — o modelo seleciona a ação errada para o intent correto
3. Parameter Error — o modelo chama a ferramenta com argumentos errados ou incompletos.

Format Error — o modelo retorna output em formato inválido (campo que deveria ser JSON vem como string, enum fora do conjunto válido). Detectável por asserção estrutural — exatamente o que seu glossário chama de functional correctness.
Invalid Action — o modelo tenta chamar uma ferramenta que não existe ou chama com nome errado. Em LangGraph isso gera exception capturável.
Tool Execution Error — a ferramenta falha e o modelo não trata o erro adequadamente. Você consegue simular injetando falhas na ferramenta mock.

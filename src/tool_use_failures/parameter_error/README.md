Quando essa falha ocorre?

1. Incorreção no Formato e na Assinatura da API

As falhas ocorrem quando o agente gera argumentos que não respeitam a "assinatura" (os requisitos técnicos) da função chamada. Isso inclui:
- Tipos de dados inválidos: Por exemplo, passar uma string quando a API espera um objeto JSON.
- Número incorreto de argumentos: O agente pode omitir parâmetros obrigatórios ou incluir argumentos extras que a ferramenta não reconhece.

2. Mapeamento Semântico e Contextual Errado

Ocorre quando o parâmetro é tecnicamente válido (ex: é uma string), mas o conteúdo está incorreto para o contexto da tarefa.
- Exemplo prático: Em um ambiente de nuvem, um agente pode tentar escalar uma instância usando o nome da região em um campo que exige o ID da instância.
- Alucinação de parâmetros: O agente pode inventar campos ou chaves que não existem na especificação da ferramenta (campos inexistentes).

3. Falhas de Construção (Chamadas Malformadas)

Muitas vezes, o agente entende o que precisa fazer, mas falha na "disciplina sintática" ao construir a chamada. Isso inclui:
- Sintaxe JSON inválida: Gerar um código com erros de pontuação ou estrutura que impede a ferramenta de ler os dados.
- Nomes de ferramentas incorretos: O agente utiliza um nome semelhante, mas não idêntico ao registrado no sistema (ex: usar database_query em vez de db_query_tool).
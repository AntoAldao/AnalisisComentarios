import openai

openai.api_key = "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"
comentarios=[
    "La Doctora zally excelente profesional, muy buena su atención se nota que su dedicación es real y completa",
    "El médico procedió correctamente.Las Enfermeras y teens muy gentiles .",
    "El tiempo que se tomó el médico que me atendió para explicarme todo con detalle.",
    "atencion mala"
]
for comentario in comentarios:
    pregunta = "El siguiente comentario es positivo o negativo o sin contexto?"+comentario
    response = openai.Completion.create(engine="text-davinci-003",
                                        prompt=pregunta, 
                                        max_tokens=4000)

    response = response.choices[0].text
    print(response)
## Analisis de relajación termal para diferentes relaciones de masa en un gas AHP

Utilizando (`gas03_analysis.py`) se estudio un gas unidimensional que interactua unicamente por colisiones elasticas para particulas de diferentes masas $m_1$ y $m_2$ situadas en un anillo periodico (condiciones periodicas). Estudiando si al establecer una temperatura inicial $T(x,0) = T_0[1 + a\cos(kx)]$ en el sistema,  si la temperatura, y otros perfiles hidrodinamicos, evoluciona como predice la hidrodinamica, con ondas de sonido (Euler linealizado) y un modo de entropía que solo decae por conducción de calor.

Esto con el objetivo de no solo estudiar el cambio en la temperatura, sino que para ver si el cambio de relación de masas cambia a la relajación termica.

Además quisimos probar que nuestro motor efectivamente simula lo que está en el paper Blast in a One Dimensional Cold Gas: From Newtonian Dynamics to Hydrodinamics. Donde, se utiliza un gas unidimensional AHP (Alternating - Hard- PArticle) para verificar que la dinamica microscopica del gas AHP se comporta con la hidrodinamica de un gas frio, estableciendo una condición inicial en la energía $E(x,0)$ y estudiando el sistema desde entonces. 

En este caso, se establecen para el gas condiciones iniciales en la energía y se analizan los perfiles hidrodinamicos de la simulación. Pudimos reproducir estos resultados utilizando nuestro motor. Como en este paper se introduce una condición inicial en la energía $E(x,0)$, se tuvo que escribir otro código, manteniendo el motor de la simulación (`ahp_engine_c.py`), para poder establecer bien la condición inicial y estudiar los resultados (`blast_ahp.py`).

Las imagenes de los analisis se encuentran en las carpetas `\figuras-gas03_r2` (r5 y r10 dependiendo de la relación de masas)y `\figuras-blast` respectivamente. 

Además, se repitio el analisis para gases con diferentes relaciones de masas $r = m_1/m_2$, los resultados de este analisis estan en `\figuras_gas03_r2_con_mass_ratios`. 



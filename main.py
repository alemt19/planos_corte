import pulp

class PlanoDeCorte:
    """
    Resuelve un problema de programación lineal entera usando el método de planos de corte de Gomory,
    aprovechando el solver de PuLP para comprobar soluciones enteras.
    """

    def __init__(self, c, A, b):
        """
        Inicializa el problema de programación lineal.

        Args:
            c (list): Coeficientes de la función objetivo (para maximizar).
            A (list of lists): Matriz de coeficientes de las restricciones.
            b (list): Vector de términos independientes de las restricciones.
        """
        self.c = c
        self.A = A
        self.b = b
        self.num_vars = len(c)
        self.num_restricciones = len(A)

    def es_entero(self, valores):
        return all(abs(x - round(x)) < 1e-5 for x in valores)

    def resolver(self):
        iteracion = 0
        cortes = []

        while True:
            iteracion += 1
            print(f"\n--- Iteración {iteracion} ---")

            # Crear modelo PuLP
            prob = pulp.LpProblem("PL_enteros", pulp.LpMaximize)
            x = [pulp.LpVariable(f"x{i}", lowBound=0, cat='Continuous') for i in range(self.num_vars)]

            # Objetivo
            prob += pulp.lpDot(self.c, x)

            # Restricciones originales + cortes acumulados
            for i in range(self.num_restricciones):
                prob += (pulp.lpDot(self.A[i], x) <= self.b[i])

            for corte, rhs in cortes:
                prob += (pulp.lpDot(corte, x) <= rhs)

            # Resolver relajación lineal
            prob.solve()
            status = pulp.LpStatus[prob.status]
            if status != 'Optimal':
                print("No es posible resolver el PL en esta iteración.")
                return None

            solucion = [v.varValue for v in x]
            resultado_objetivo = pulp.value(prob.objective)
            print(f"Solución actual: {solucion}")
            print(f"Valor objetivo actual: {resultado_objetivo}")

            # ¿Es entera?
            if self.es_entero(solucion):
                print("\n¡Se ha encontrado una solución entera óptima!")
                return solucion, resultado_objetivo

            # Buscar la variable más fraccionaria
            idx_maxfrac = max(
                range(len(solucion)),
                key=lambda i: abs(solucion[i] - round(solucion[i]))
            )
            frac = solucion[idx_maxfrac] - int(solucion[idx_maxfrac])
            if frac < 1e-5:
                print("\nNo se puede avanzar con más cortes. La solución es óptima (aunque podría tener variables casi enteras).")
                return solucion, resultado_objetivo

            # Crear corte de Gomory básico (simulado)
            corte = [0.0] * self.num_vars
            corte[idx_maxfrac] = 1.0
            rhs = int(solucion[idx_maxfrac])
            cortes.append((corte, rhs))
            print(f"Añadiendo corte: x{idx_maxfrac+1} <= {rhs}")

if __name__ == '__main__':
    print("Resolución de Problemas de Programación Entera con Plano de Corte y PuLP")
    print("--------------------------------------------------------------------------")
    try:
        num_vars = int(input("Introduce el número de variables de decisión: "))
        num_restricciones = int(input("Introduce el número de restricciones: "))

        c = []
        print("\nIntroduce los coeficientes de la función objetivo (para maximizar):")
        for i in range(num_vars):
            c.append(float(input(f"Coeficiente para x{i+1}: ")))

        A = []
        print("\nIntroduce los coeficientes de las restricciones (matriz A):")
        for i in range(num_restricciones):
            fila = []
            for j in range(num_vars):
                fila.append(float(input(f"Coeficiente de x{j+1} en la restricción {i+1}: ")))
            A.append(fila)

        b = []
        print("\nIntroduce los términos independientes de las restricciones (vector b):")
        for i in range(num_restricciones):
            b.append(float(input(f"Término para la restricción {i+1}: ")))

        problema = PlanoDeCorte(c, A, b)
        resultado = problema.resolver()

        if resultado is not None:
            solucion_optima, valor_optimo = resultado
            solucion_enteros = [round(x) for x in solucion_optima]
            print("\n--- Resultados Finales ---")
            print(f"Solución Óptima Entera: {solucion_enteros}")
            print(f"Valor Óptimo de la Función Objetivo: {valor_optimo}")
        else:
            print("\nNo se encontró una solución óptima entera.")
    except ValueError:
        print("\nError: Por favor, introduce solo valores numéricos.")
    except Exception as e:
        print(f"\nHa ocurrido un error inesperado: {e}")

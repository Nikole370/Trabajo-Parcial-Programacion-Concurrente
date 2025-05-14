package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"strconv"
	"time"
)

// IMPLEMENTACIÓN SECUENCIAL DE LA REGRESIÓN LOGÍSTICA

// Función sigmoide
func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

// Función para calcular la predicción
func predict(X []float64, weights []float64) float64 {
	var z float64
	for i := 0; i < len(X); i++ {
		z += X[i] * weights[i]
	}
	return sigmoid(z)
}

// Entrenamiento con gradiente descendente
func train(X [][]float64, y []float64, learningRate float64, iterations int) []float64 {
	features := len(X[0])
	weights := make([]float64, features)

	for iter := 0; iter < iterations; iter++ {
		gradients := make([]float64, features)

		for i := 0; i < len(X); i++ {
			pred := predict(X[i], weights)
			error := pred - y[i]

			for j := 0; j < features; j++ {
				gradients[j] += error * X[i][j]
			}
		}

		// Actualizar pesos
		for j := 0; j < features; j++ {
			weights[j] -= learningRate * gradients[j] / float64(len(X))
		}
	}
	return weights
}

// Leer el dataset CSV
func loadCSVData(path string) ([][]float64, []float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	var X [][]float64
	var y []float64

	// Saltar encabezado
	for i, row := range records {
		if i == 0 {
			continue
		}
		// Rating
		rating, _ := strconv.ParseFloat(row[4], 64)
		// Número de reseñas
		numReviews, _ := strconv.ParseFloat(row[5], 64)

		// Ejemplo con 2 features + bias
		xi := []float64{1, rating, numReviews}
		X = append(X, xi)

		// Etiqueta: 1 si rating >= 4.0
		label := 0.0
		if rating >= 4.0 {
			label = 1.0
		}
		y = append(y, label)
	}

	return X, y, nil
}

func main() {
	X, y, err := loadCSVData("yelp_database.csv")
	if err != nil {
		fmt.Println("Error al cargar datos:", err)
		return
	}

	learningRate := 0.1
	iterations := 1000

	start := time.Now()
	weights := train(X, y, learningRate, iterations)
	fmt.Println("Pesos entrenados:", weights)

	// Prueba
	muestra := []float64{1, 4.2, 120} // Rating 4.2, 120 reviews
	prob := predict(muestra, weights)
	fmt.Printf("Probabilidad de clase 1: %.4f\n", prob)
	fmt.Println("Tiempo de ejecución:", time.Since(start))
}

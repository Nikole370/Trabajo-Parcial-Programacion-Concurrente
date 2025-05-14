package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"strconv"
	"time"
)

// ----------- Funciones comunes -----------

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func predict(X []float64, weights []float64) float64 {
	var z float64
	for i := 0; i < len(X); i++ {
		z += X[i] * weights[i]
	}
	return sigmoid(z)
}

func loadCSVData(path string) ([][]float64, []float64, float64, float64, float64, float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, 0, 0, 0, 0, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, 0, 0, 0, 0, err
	}

	var X [][]float64
	var y []float64
	minRating, maxRating := math.MaxFloat64, -math.MaxFloat64
	minReviews, maxReviews := math.MaxFloat64, -math.MaxFloat64

	for i, row := range records {
		if i == 0 {
			continue
		}
		rating, err1 := strconv.ParseFloat(row[5], 64)
		numReviews, err2 := strconv.ParseFloat(row[6], 64)
		if err1 != nil || err2 != nil {
			continue
		}

		if rating < minRating {
			minRating = rating
		}
		if rating > maxRating {
			maxRating = rating
		}
		if numReviews < minReviews {
			minReviews = numReviews
		}
		if numReviews > maxReviews {
			maxReviews = numReviews
		}

		xi := []float64{1, rating, numReviews}
		X = append(X, xi)

		label := 0.0
		if rating >= 4.0 {
			label = 1.0
		}
		y = append(y, label)
	}
	return X, y, minRating, maxRating, minReviews, maxReviews, nil
}

func normalizeFeatures(X [][]float64, minRating, maxRating, minReviews, maxReviews float64) {
	for i := 0; i < len(X); i++ {
		X[i][1] = (X[i][1] - minRating) / (maxRating - minRating)
		X[i][2] = (X[i][2] - minReviews) / (maxReviews - minReviews)
	}
}

// ----------- Entrenamiento Secuencial -----------

func trainSequential(X [][]float64, y []float64, learningRate float64, iterations int) []float64 {
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
		for j := 0; j < features; j++ {
			weights[j] -= learningRate * gradients[j] / float64(len(X))
		}
	}
	return weights
}

func calculateAccuracy(X [][]float64, y []float64, weights []float64) float64 {
	correct := 0
	for i := 0; i < len(X); i++ {
		pred := predict(X[i], weights)
		if (pred >= 0.5 && y[i] == 1.0) || (pred < 0.5 && y[i] == 0.0) {
			correct++
		}
	}
	return float64(correct) / float64(len(X)) * 100
}

func main() {
	// Cargar datos desde el CSV
	X, y, minRating, maxRating, minReviews, maxReviews, err := loadCSVData("yelp_database.csv")
	if err != nil {
		fmt.Println("Error al cargar datos:", err)
		return
	}
	normalizeFeatures(X, minRating, maxRating, minReviews, maxReviews)

	// Configuración de parámetros
	learningRate := 0.1
	iterations := 1000

	// Entrenamiento secuencial
	start := time.Now()
	weights := trainSequential(X, y, learningRate, iterations)
	duration := time.Since(start)

	// Cálculo de precisión
	accuracy := calculateAccuracy(X, y, weights)

	// Mostrar resultados
	fmt.Println("--- Modo Secuencial ---")
	fmt.Printf("Pesos: %v\n", weights)
	fmt.Printf("Precisión: %.2f%%\n", accuracy)
	fmt.Printf("Tiempo de ejecución: %v\n", duration)
}

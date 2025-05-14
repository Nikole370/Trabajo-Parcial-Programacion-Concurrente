package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
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

// ----------- Entrenamiento secuencial -----------

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

// ----------- Entrenamiento concurrente optimizado con minibatches -----------

func trainConcurrent(X [][]float64, y []float64, learningRate float64, iterations int, batchSize int) []float64 {
	features := len(X[0])
	weights := make([]float64, features)

	for iter := 0; iter < iterations; iter++ {
		gradients := make([]float64, features)
		var wg sync.WaitGroup
		batches := len(X) / batchSize
		if len(X)%batchSize != 0 {
			batches++
		}

		// Procesamos minibatches
		for b := 0; b < batches; b++ {
			wg.Add(1)
			go func(batchIndex int) {
				defer wg.Done()
				start := batchIndex * batchSize
				end := start + batchSize
				if end > len(X) {
					end = len(X)
				}
				
				partialGradients := make([]float64, features)
				for i := start; i < end; i++ {
					pred := predict(X[i], weights)
					error := pred - y[i]
					for j := 0; j < features; j++ {
						partialGradients[j] += error * X[i][j]
					}
				}
				
				// Acumulamos gradientes globales
				for j := 0; j < features; j++ {
					gradients[j] += partialGradients[j]
				}
			}(b)
		}

		// Esperamos que todas las goroutines terminen
		wg.Wait()

		// Actualizamos los pesos después de cada iteración
		for j := 0; j < features; j++ {
			weights[j] -= learningRate * gradients[j] / float64(len(X))
		}
	}
	return weights
}

// Función para calcular la precisión

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

// ----------- Menú principal -----------

func main() {
	X, y, minRating, maxRating, minReviews, maxReviews, err := loadCSVData("yelp_database.csv")
	if err != nil {
		fmt.Println("Error al cargar datos:", err)
		return
	}
	normalizeFeatures(X, minRating, maxRating, minReviews, maxReviews)

	learningRate := 0.1
	iterations := 1000
	batchSize := 100 // Tamaño del minibatch

	// Normalizar muestra manualmente
	rawMuestra := []float64{1, 4.2, 120}
	rawMuestra[1] = (rawMuestra[1] - minRating) / (maxRating - minRating)
	rawMuestra[2] = (rawMuestra[2] - minReviews) / (maxReviews - minReviews)

	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Println("\n======= MENÚ =======")
		fmt.Println("1. Entrenar (Secuencial)")
		fmt.Println("2. Entrenar (Concurrente)")
		fmt.Println("3. Comparar ambos")
		fmt.Println("4. Salir")
		fmt.Print("Seleccione una opción: ")

		input, _ := reader.ReadString('\n')
		choice := strings.TrimSpace(input)

		switch choice {
		case "1":
			start := time.Now()
			weights := trainSequential(X, y, learningRate, iterations)
			duration := time.Since(start)
			accuracy := calculateAccuracy(X, y, weights)
			fmt.Println("\n--- Modo Secuencial ---")
			fmt.Println("Pesos:", weights)
			fmt.Printf("Probabilidad ejemplo: %.4f\n", predict(rawMuestra, weights))
			fmt.Printf("Precisión: %.2f%%\n", accuracy)
			fmt.Println("Tiempo:", duration)
		case "2":
			start := time.Now()
			weights := trainConcurrent(X, y, learningRate, iterations, batchSize)
			duration := time.Since(start)
			accuracy := calculateAccuracy(X, y, weights)
			fmt.Println("\n--- Modo Concurrente ---")
			fmt.Println("Pesos:", weights)
			fmt.Printf("Probabilidad ejemplo: %.4f\n", predict(rawMuestra, weights))
			fmt.Printf("Precisión: %.2f%%\n", accuracy)
			fmt.Println("Tiempo:", duration)
		case "3":
			startSeq := time.Now()
			weightsSeq := trainSequential(X, y, learningRate, iterations)
			durSeq := time.Since(startSeq)
			accuracySeq := calculateAccuracy(X, y, weightsSeq)

			startConc := time.Now()
			weightsConc := trainConcurrent(X, y, learningRate, iterations, batchSize)
			durConc := time.Since(startConc)
			accuracyConc := calculateAccuracy(X, y, weightsConc)

			fmt.Println("\n--- Comparación ---")
			fmt.Printf("Secuencial: Tiempo: %v | Precisión: %.2f%% | Probabilidad: %.4f\n", durSeq, accuracySeq, predict(rawMuestra, weightsSeq))
			fmt.Printf("Concurrente: Tiempo: %v | Precisión: %.2f%% | Probabilidad: %.4f\n", durConc, accuracyConc, predict(rawMuestra, weightsConc))
		case "4":
			fmt.Println("Saliendo...")
			return
		default:
			fmt.Println("Opción inválida.")
		}
	}
}
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

func loadCSVData(path string) (trainX, testX [][]float64, trainY, testY []float64, minRating, maxRating, minReviews, maxReviews float64, err error) {
	file, err := os.Open(path)
	if err != nil {
		return
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return
	}

	var allX [][]float64
	var allY []float64
	minRating, maxRating = math.MaxFloat64, -math.MaxFloat64
	minReviews, maxReviews = math.MaxFloat64, -math.MaxFloat64

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
		allX = append(allX, xi)

		label := 0.0
		if rating >= 4.0 {
			label = 1.0
		}
		allY = append(allY, label)
	}

	// Dividir 80/20
	total := len(allX)
	split := int(0.8 * float64(total))
	for i := 0; i < split; i++ {
		trainX = append(trainX, allX[i])
		trainY = append(trainY, allY[i])
	}
	for i := split; i < total; i++ {
		testX = append(testX, allX[i])
		testY = append(testY, allY[i])
	}

	return
}

func normalizeFeatures(X [][]float64, minRating, maxRating, minReviews, maxReviews float64) {
	for i := 0; i < len(X); i++ {
		X[i][1] = (X[i][1] - minRating) / (maxRating - minRating)
		X[i][2] = (X[i][2] - minReviews) / (maxReviews - minReviews)
	}
}

// ----------- Entrenamiento secuencial -----------

func trainSequential(X [][]float64, y []float64, learningRate float64, iterations int, batchSize int) []float64 {
	features := len(X[0])
	weights := make([]float64, features)
	dataLen := len(X)

	for iter := 0; iter < iterations; iter++ {
		for i := 0; i < dataLen; i += batchSize {
			end := i + batchSize
			if end > dataLen {
				end = dataLen
			}

			gradients := make([]float64, features)
			for j := i; j < end; j++ {
				pred := predict(X[j], weights)
				error := pred - y[j]
				for k := 0; k < features; k++ {
					gradients[k] += error * X[j][k]
				}
			}

			for k := 0; k < features; k++ {
				weights[k] -= learningRate * gradients[k] / float64(end-i)
			}
		}
	}
	return weights
}

// ----------- Entrenamiento concurrente optimizado con minibatches -----------
func trainConcurrent(X [][]float64, y []float64, learningRate float64, iterations int, batchSize int) []float64 {
	features := len(X[0])
	weights := make([]float64, features)
	dataLen := len(X)

	for iter := 0; iter < iterations; iter++ {
		var wg sync.WaitGroup
		var mutex sync.Mutex

		// Recorremos los mini-batches como en la secuencial
		for i := 0; i < dataLen; i += batchSize {
			wg.Add(1)

			// Capturar valores para la goroutine
			start := i
			end := i + batchSize
			if end > dataLen {
				end = dataLen
			}

			go func(start, end int) {
				defer wg.Done()
				partialGradients := make([]float64, features)

				for j := start; j < end; j++ {
					pred := predict(X[j], weights)
					error := pred - y[j]
					for k := 0; k < features; k++ {
						partialGradients[k] += error * X[j][k]
					}
				}

				mutex.Lock()
				for k := 0; k < features; k++ {
					weights[k] -= learningRate * partialGradients[k] / float64(end-start)
				}
				mutex.Unlock()
			}(start, end)
		}

		wg.Wait()
	}
	return weights
}

func trimmedMean(times []float64, trimCount int) float64 {
	if len(times) <= 2*trimCount {
		panic("No hay suficientes datos para calcular la media recortada.")
	}
	sort.Float64s(times)
	trimmed := times[trimCount : len(times)-trimCount]

	var sum float64
	for _, t := range trimmed {
		sum += t
	}
	return sum / float64(len(trimmed))
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
	trainX, testX, trainY, testY, minRating, maxRating, minReviews, maxReviews, err := loadCSVData("yelp_database.csv")
	if err != nil {
		fmt.Println("Error al cargar datos:", err)
		return
	}
	normalizeFeatures(trainX, minRating, maxRating, minReviews, maxReviews)
	normalizeFeatures(testX, minRating, maxRating, minReviews, maxReviews)

	learningRate := 0.1
	iterations := 750
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
		fmt.Println("4. Benchmark")
		fmt.Println("5. Salir")
		fmt.Print("Seleccione una opción: ")

		input, _ := reader.ReadString('\n')
		choice := strings.TrimSpace(input)

		switch choice {
		case "1":
			start := time.Now()
			weights := trainSequential(trainX, trainY, learningRate, iterations, batchSize)
			duration := time.Since(start)
			accuracy := calculateAccuracy(testX, testY, weights)
			fmt.Println("\n--- Modo Secuencial ---")
			fmt.Println("Pesos:", weights)
			fmt.Printf("Probabilidad ejemplo: %.4f\n", predict(rawMuestra, weights))
			fmt.Printf("Precisión (test): %.2f%%\n", accuracy)
			fmt.Println("Tiempo:", duration)
		case "2":
			start := time.Now()
			weights := trainConcurrent(trainX, trainY, learningRate, iterations, batchSize)
			duration := time.Since(start)
			accuracy := calculateAccuracy(testX, testY, weights)
			fmt.Println("\n--- Modo Concurrente ---")
			fmt.Println("Pesos:", weights)
			fmt.Printf("Probabilidad ejemplo: %.4f\n", predict(rawMuestra, weights))
			fmt.Printf("Precisión (test): %.2f%%\n", accuracy)
			fmt.Println("Tiempo:", duration)
		case "3":
			startSeq := time.Now()
			weightsSeq := trainSequential(trainX, trainY, learningRate, iterations, batchSize)
			durSeq := time.Since(startSeq)
			accuracySeq := calculateAccuracy(testX, testY, weightsSeq)

			startConc := time.Now()
			weightsConc := trainConcurrent(trainX, trainY, learningRate, iterations, batchSize)
			durConc := time.Since(startConc)
			accuracyConc := calculateAccuracy(testX, testY, weightsConc)

			fmt.Println("\n--- Comparación ---")
			fmt.Printf("Secuencial: Tiempo: %v | Precisión (test): %.2f%% | Probabilidad: %.4f\n",
				durSeq, accuracySeq, predict(rawMuestra, weightsSeq))
			fmt.Printf("Concurrente: Tiempo: %v | Precisión (test): %.2f%% | Probabilidad: %.4f\n",
				durConc, accuracyConc, predict(rawMuestra, weightsConc))
		case "4":
			const total = 1000
			var timesSeq []float64
			var timesConc []float64

			fmt.Println("\n--- Iniciando benchmark de 1000 repeticiones ---")
			for i := 0; i < total; i++ {
				startSeq := time.Now()
				_ = trainSequential(trainX, trainY, learningRate, iterations, batchSize)
				timesSeq = append(timesSeq, time.Since(startSeq).Seconds())

				startConc := time.Now()
				_ = trainConcurrent(trainX, trainY, learningRate, iterations, batchSize)
				timesConc = append(timesConc, time.Since(startConc).Seconds())

				// Mostrar progreso cada 10 iteraciones
				if (i+1)%10 == 0 || i == total-1 {
					percent := float64(i+1) / float64(total) * 100
					fmt.Printf("\rProgreso: %4.1f%% [%d/%d]", percent, i+1, total)
				}
			}
			fmt.Println() // salto de línea

			meanSeq := trimmedMean(timesSeq, 50)
			meanConc := trimmedMean(timesConc, 50)

			fmt.Println("\n--- Benchmark completado ---")
			fmt.Printf("Media recortada Secuencial (seg): %.4f\n", meanSeq)
			fmt.Printf("Media recortada Concurrente (seg): %.4f\n", meanConc)
		case "5":
			fmt.Println("Saliendo...")
			return
		default:
			fmt.Println("Opción inválida.")
		}
	}
}

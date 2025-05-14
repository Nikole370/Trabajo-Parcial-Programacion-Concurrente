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

	for i, row := range records {
		if i == 0 {
			continue
		}
		rating, err1 := strconv.ParseFloat(row[5], 64)
		numReviews, err2 := strconv.ParseFloat(row[6], 64)
		if err1 != nil || err2 != nil {
			continue
		}

		xi := []float64{1, rating, numReviews}
		X = append(X, xi)

		label := 0.0
		if rating >= 4.0 {
			label = 1.0
		}
		y = append(y, label)
	}
	return X, y, nil
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

// ----------- Entrenamiento concurrente -----------

func trainConcurrent(X [][]float64, y []float64, learningRate float64, iterations int) []float64 {
	features := len(X[0])
	weights := make([]float64, features)

	for iter := 0; iter < iterations; iter++ {
		gradients := make([]float64, features)
		var mutex sync.Mutex
		var wg sync.WaitGroup

		for i := 0; i < len(X); i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				pred := predict(X[i], weights)
				error := pred - y[i]
				localGradient := make([]float64, features)
				for j := 0; j < features; j++ {
					localGradient[j] = error * X[i][j]
				}
				mutex.Lock()
				for j := 0; j < features; j++ {
					gradients[j] += localGradient[j]
				}
				mutex.Unlock()
			}(i)
		}

		wg.Wait()

		for j := 0; j < features; j++ {
			weights[j] -= learningRate * gradients[j] / float64(len(X))
		}
	}
	return weights
}

// ----------- Menú principal -----------

func main() {
	X, y, err := loadCSVData("yelp_database.csv")
	if err != nil {
		fmt.Println("Error al cargar el archivo:", err)
		return
	}

	learningRate := 0.1
	iterations := 1000
	muestra := []float64{1, 4.2, 120}

	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Println("\n======= MENÚ =======")
		fmt.Println("1. Entrenar regresión logística (Secuencial)")
		fmt.Println("2. Entrenar regresión logística (Concurrente)")
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
			fmt.Println("\n--- Modo Secuencial ---")
			fmt.Println("Pesos:", weights)
			fmt.Printf("Probabilidad ejemplo: %.4f\n", predict(muestra, weights))
			fmt.Println("Tiempo:", duration)

		case "2":
			start := time.Now()
			weights := trainConcurrent(X, y, learningRate, iterations)
			duration := time.Since(start)
			fmt.Println("\n--- Modo Concurrente ---")
			fmt.Println("Pesos:", weights)
			fmt.Printf("Probabilidad ejemplo: %.4f\n", predict(muestra, weights))
			fmt.Println("Tiempo:", duration)

		case "3":
			startSeq := time.Now()
			weightsSeq := trainSequential(X, y, learningRate, iterations)
			durSeq := time.Since(startSeq)

			startConc := time.Now()
			weightsConc := trainConcurrent(X, y, learningRate, iterations)
			durConc := time.Since(startConc)

			fmt.Println("\n--- Comparación ---")
			fmt.Println("Secuencial:")
			fmt.Printf("Tiempo: %v | Probabilidad: %.4f\n", durSeq, predict(muestra, weightsSeq))
			fmt.Println("Concurrente:")
			fmt.Printf("Tiempo: %v | Probabilidad: %.4f\n", durConc, predict(muestra, weightsConc))

		case "4":
			fmt.Println("Saliendo...")
			return

		default:
			fmt.Println("Opción inválida.")
		}
	}
}

//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include "word_center.h"

void print_vector(float *vector, long long dimensionality) {
    printf("[");
    for (unsigned int i = 0; i < dimensionality; i++) {
        if (i == dimensionality - 1)
            printf("%f]", vector[i]);
        else
            printf("%f, ", vector[i]);
    }
}

float *get_model() { return model; }
char *get_dictionary() { return dictionary; }
long long get_dimensionality() { return dimensionality; }
long long get_dictionary_size() { return dictionary_size; }

int load_model(char *file_name) {
    if (model != NULL || dictionary != NULL) {
        printf("Model already loaded\n");
        return -1;
    }
    // Load model from file
    FILE *file_pointer;

    // Open file
    file_pointer = fopen(file_name, "rb");
    if (file_pointer == NULL) {
      printf("Input file not found\n");
      return -1;
    }

    // Get number of words in dictionary
    fscanf(file_pointer, "%lld", &dictionary_size);
    // Get dimensionality of word vectors
    fscanf(file_pointer, "%lld", &dimensionality);
    // Holds dictionary
    dictionary = (char *)malloc((long long) dictionary_size * MAX_WORD_LENGTH * sizeof(char));
    if (dictionary == NULL) {
        printf("Cannot allocate memory: %lld MB\n",
            (long long) dictionary_size * MAX_WORD_LENGTH * sizeof(char) / 1048576);
        return -1;
    }
    // Allocate buffer for model
    model = (float *)malloc((long long) dictionary_size * (long long) dimensionality * sizeof(float));
    if (model == NULL) {
        printf("Cannot allocate memory: %lld MB\n",
            (long long) dictionary_size * dimensionality * sizeof(float) / 1048576);
        free(dictionary);
        return -1;
    }

    printf("Loading model...\n");
    for (long long i = 0; i < dictionary_size; i++) {
        long long j = 0;
        // Copy words to dictionary
        while (1) {
            dictionary[i * MAX_WORD_LENGTH + j] = fgetc(file_pointer);
            if (feof(file_pointer) || (dictionary[i * MAX_WORD_LENGTH + j] == ' '))
                break;
            if ((j < MAX_WORD_LENGTH) && (dictionary[i * MAX_WORD_LENGTH + j] != '\n'))
                j++;
        }
        dictionary[i * MAX_WORD_LENGTH + j] = 0;
        // Copy vector representation of word to model
        for (j = 0; j < dimensionality; j++)
          fread(&model[j + i * dimensionality], sizeof(float), 1, file_pointer);
    }
    fclose(file_pointer);
    printf("Successfully loaded %lld vectors with %lld dimensions\n", dictionary_size, dimensionality);
    return 0;
}

void free_model() {
    if (model != NULL) {
        free(model);
        model = NULL;
    }
    if (dictionary != NULL) {
        free(dictionary);
        dictionary = NULL;
    }
    if (word_center != NULL) {
        free(word_center);
        word_center = NULL;
    }
}

float *compute_center(char *words, unsigned int num_words) {
    if (model == NULL || dictionary == NULL) {
        printf("Model not loaded\n");
        return NULL;
    }

    // Compute center of given words
    long long pos_in_dict[num_words];
    unsigned int valid_words = num_words;
    if (word_center == NULL)
        word_center = (float *)malloc((long long) dimensionality * sizeof(float));
    // Find each word in dictionary and store position in buffer
    for (unsigned int i = 0; i < num_words; i++) {
        long long j;
        for (j = 0; j < dictionary_size; j++) {
            if (!strcmp(&dictionary[j * MAX_WORD_LENGTH],
                        &words[i * MAX_WORD_LENGTH]))
                break;
        }
        // Word not in dictionary
        if (j == dictionary_size) {
            j = -1;
            valid_words--;
        }
        pos_in_dict[i] = j;
        // printf("%s: ", &dictionary[j * MAX_WORD_LENGTH]);
        // print_vector(&model[j * dimensionality], dimensionality);
        // printf("\n");
    }

    // Write average of word vectors to result vector
    for (unsigned int i = 0; i < dimensionality; i++) {
        float sum = 0;
        for (unsigned int j = 0; j < num_words; j++) {
            // printf("i=%d, j=%d\n", i, j);
            if (pos_in_dict[j] < 0)
                continue;
            sum += model[i + pos_in_dict[j] * dimensionality];
        }
        word_center[i] = sum / valid_words;
    }

    // Find closest word to computed center and print if
    float min_distance = 999999;
    long long closest_word = -1;
    for (long long i = 0; i < dictionary_size; i++) {
        long long j = 0;
        // Skip words from input
        for (j = 0; j < num_words; j++) {
            if (pos_in_dict[j] == i) {
                j = -1;
                break;
            }
        }
        if (j == -1)
            continue;

        // Compute distance
        float distance = 0;
        for (j = 0; j < dimensionality; j++)
            distance += ((word_center[j] - model[j + i * dimensionality]) *
                        (word_center[j] - model[j + i * dimensionality]));
        // Replace closer match with current best match
        if (min_distance > distance) {
            min_distance = distance;
            closest_word = i;
        }
    }
    if (closest_word >= 0) {
        printf("Closest word to computed center is %s: ",
            &dictionary[closest_word * MAX_WORD_LENGTH]);
        print_vector(&model[closest_word * dimensionality], dimensionality);
        printf("\n");
    }
    return word_center;
}

// int main(int argc, char **argv) {
//     char file_name[MAX_PATH_LENGTH];
//     if (argc < 3) {
//         printf("Usage: ./distance <FILE> <WORDS>\nwhere FILE contains word projections in the BINARY FORMAT\n");
//         return 0;
//     }
//     // Read input
//     strcpy(file_name, argv[1]);
//     unsigned int number_of_words = argc - 2;
//     char words_to_center[MAX_WORD_LENGTH * number_of_words];
//     for (unsigned int i = 0; i < number_of_words; i++) {
//         strcpy(&words_to_center[i * MAX_WORD_LENGTH], argv[i+2]);
//     }
//
//     load_model(file_name);
//
//     // printf("Computing center of words ");
//     // for (unsigned int i = 0; i < number_of_words; i++) {
//     //     printf("%s ", &words_to_center[number_of_words*i]);
//     // }
//     // printf("\n");
//
//     float *word_center = compute_center(words_to_center, number_of_words);
//
//     free(word_center);
//     return 0;
// }

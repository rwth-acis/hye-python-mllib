// max length of vocabulary entries
const long long MAX_WORD_LENGTH = 50;
const unsigned int MAX_PATH_LENGTH = 2000;

long long dictionary_size;
long long dimensionality;
char *dictionary;
float *model, *word_center;

int load_model(char* file_name);
void free_model();
void print_vector(float *vector, long long dimensionality);
float *compute_center(char *words, unsigned int num_words);
float *get_model();
char *get_dictionary();
long long get_dimensionality();
long long get_dictionary_size();


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>


#define DEBUG_LAYER 0


/*  LayerType
 */
typedef enum _LayerType {
    LAYER_INPUT = 0,
    LAYER_FULL,
    LAYER_CONV
} LayerType;


/*  Layer
 */
typedef struct _Layer {

    int lid;                    /* Layer ID */
    struct _Layer* lprev;       /* Previous Layer */
    struct _Layer* lnext;       /* Next Layer */

    int depth, width, height;   /* Shape */

    int nnodes;                 /* Num. of Nodes */
    double* outputs;            /* Node Outputs */
    double* gradients;          /* Node Gradients */
    double* errors;             /* Node Errors */

    int nbiases;                /* Num. of Biases */
    double* biases;             /* Biases (trained) */
    double* u_biases;           /* Bias updates */

    int nweights;               /* Num. of Weights */
    double* weights;            /* Weights (trained) */
    double* u_weights;          /* Weight updates */

    LayerType ltype;            /* Layer type */
    union {
        /* Full */
        struct {
        } full;

        /* Conv */
        struct {
            int kernsize;       /* kernel size (>0) */
            int padding;        /* padding size */
            int stride;         /* stride (>0) */
        } conv;
    };

} Layer;



/*  Self made functions
 */

/* rnd(): uniform random [0.0, 1.0] */
static inline double rnd()
{
    return ((double)rand() / RAND_MAX);
}

/* nrnd(): normal random (std=1.0) */
static inline double nrnd()
{
    return (rnd()+rnd()+rnd()+rnd()-2.0) * 1.724; /* std=1.0 */
}

#if 0
/* sigmoid(x): sigmoid function */
static inline double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}
/* sigmoid_d(y): sigmoid gradient */
static inline double sigmoid_g(double y)
{
    return y * (1.0 - y);
}
#endif

#if 0
/* tanh(x): hyperbolic tangent */
static inline double tanh(double x)
{
    return 2.0 / (1.0 + exp(-2*x)) - 1.0;
}
#endif
/* tanh_g(y): hyperbolic tangent gradient */
static inline double tanh_g(double y)
{
    return 1.0 - y*y;
}

/* relu(x): ReLU */
static inline double relu(double x)
{
    return (0 < x)? x : 0;
}
/* relu_g(y): ReLU gradient */
static inline double relu_g(double y)
{
    return (0 < y)? 1 : 0;
}

double myExp(double x) {
    double sum = 1.0;
    double term = 1.0;

    for (int n = 1; n < 10; ++n) {
        term *= x / n;
        sum += term;
    }

    return sum;
}

double myTanh(double x) {
    if (x < -3.0) {
        return -1.0;
    } else if (x > 3.0) {
        return 1.0;
    } else {
        double x2 = x * x;
        double term = x;
        double sum = x;

        for (int n = 1; n < 10; ++n) {
            term *= x2;
            sum += term / ((2 * n + 1) * (2 * n + 1));
        }

        return sum;
    }
}








/*  Layer
 */

/* Layer_create(lprev, ltype, depth, width, height, nbiases, nweights)
   Creates a Layer object for internal use.
*/
static Layer* Layer_create(
    Layer* lprev, LayerType ltype,
    int depth, int width, int height,
    int nbiases, int nweights)
{
    Layer* self = (Layer*)calloc(1, sizeof(Layer));
    if (self == NULL) return NULL;

    self->lprev = lprev;
    self->lnext = NULL;
    self->ltype = ltype;
    self->lid = 0;
    if (lprev != NULL) {
        assert (lprev->lnext == NULL);
        lprev->lnext = self;
        self->lid = lprev->lid+1;
    }
    self->depth = depth;
    self->width = width;
    self->height = height;

    /* Nnodes: number of outputs. */
    self->nnodes = depth * width * height;
    self->outputs = (double*)calloc(self->nnodes, sizeof(double));
    self->gradients = (double*)calloc(self->nnodes, sizeof(double));
    self->errors = (double*)calloc(self->nnodes, sizeof(double));

    self->nbiases = nbiases;
    self->biases = (double*)calloc(self->nbiases, sizeof(double));
    self->u_biases = (double*)calloc(self->nbiases, sizeof(double));

    self->nweights = nweights;
    self->weights = (double*)calloc(self->nweights, sizeof(double));
    self->u_weights = (double*)calloc(self->nweights, sizeof(double));

    return self;
}

/* Layer_destroy(self)
   Releases the memory.
*/
void Layer_destroy(Layer* self)
{
    assert (self != NULL);

    free(self->outputs);
    free(self->gradients);
    free(self->errors);

    free(self->biases);
    free(self->u_biases);
    free(self->weights);
    free(self->u_weights);

    free(self);
}

/* Layer_dump(self, fp)
   Shows the debug output.
*/
void Layer_dump(const Layer* self, FILE* fp)
{
    assert (self != NULL);
    Layer* lprev = self->lprev;
    fprintf(fp, "Layer%d ", self->lid);
    if (lprev != NULL) {
        fprintf(fp, "(lprev=Layer%d) ", lprev->lid);
    }
    fprintf(fp, "shape=(%d,%d,%d), nodes=%d\n",
            self->depth, self->width, self->height, self->nnodes);
    {
        int i = 0;
        for (int z = 0; z < self->depth; z++) {
            fprintf(fp, "  %d:\n", z);
            for (int y = 0; y < self->height; y++) {
                fprintf(fp, "    [");
                for (int x = 0; x < self->width; x++) {
                    fprintf(fp, " %.4f", self->outputs[i++]);
                }
                fprintf(fp, "]\n");
            }
        }
    }

    switch (self->ltype) {
    case LAYER_FULL:
        /* Fully connected layer. */
        assert (lprev != NULL);
        fprintf(fp, "  biases = [");
        for (int i = 0; i < self->nnodes; i++) {
            fprintf(fp, " %.4f", self->biases[i]);
        }
        fprintf(fp, "]\n");
        fprintf(fp, "  weights = [\n");
        {
            int k = 0;
            for (int i = 0; i < self->nnodes; i++) {
                fprintf(fp, "    [");
                for (int j = 0; j < lprev->nnodes; j++) {
                    fprintf(fp, " %.4f", self->weights[k++]);
                }
                fprintf(fp, "]\n");
            }
        }
        fprintf(fp, "  ]\n");
        break;

    case LAYER_CONV:
        /* Convolutional layer. */
        assert (lprev != NULL);
        fprintf(fp, "  stride=%d, kernsize=%d\n",
                self->conv.stride, self->conv.kernsize);
        {
            int k = 0;
            for (int z = 0; z < self->depth; z++) {
                fprintf(fp, "  %d: bias=%.4f, weights = [", z, self->biases[z]);
                for (int j = 0; j < lprev->depth * self->conv.kernsize * self->conv.kernsize; j++) {
                    fprintf(fp, " %.4f", self->weights[k++]);
                }
                fprintf(fp, "]\n");
            }
        }
        break;

    default:
        break;
    }
}

/* Layer_feedForw_full(self)
   Performs feed forward updates.
*/
static void Layer_feedForw_full(Layer* self)
{
    assert (self->ltype == LAYER_FULL);
    assert (self->lprev != NULL);
    Layer* lprev = self->lprev;

    int k = 0;
    for (int i = 0; i < self->nnodes; i++) {
        /* Compute Y = (W * X + B) without activation function. */
        double x = self->biases[i];
        for (int j = 0; j < lprev->nnodes; j++) {
            x += (lprev->outputs[j] * self->weights[k++]);
        }
        self->outputs[i] = x;
    }

    if (self->lnext == NULL) {
        /* Last layer - use Softmax. */
        double m = -1;
        for (int i = 0; i < self->nnodes; i++) {
            double x = self->outputs[i];
            if (m < x) { m = x; }
        }
        double t = 0;
        for (int i = 0; i < self->nnodes; i++) {
            double x = self->outputs[i];
            double y = myExp(x-m);
            self->outputs[i] = y;
            t += y;
        }
        for (int i = 0; i < self->nnodes; i++) {
            self->outputs[i] /= t;
            /* This isn't right, but set the same value to all the gradients. */
            self->gradients[i] = 1;
        }
    } else {
        /* Otherwise, use Tanh. */
        for (int i = 0; i < self->nnodes; i++) {
            double x = self->outputs[i];
            double y = myTanh(x);
            self->outputs[i] = y;
            self->gradients[i] = tanh_g(y);
        }
    }

#if DEBUG_LAYER
    fprintf(stderr, "Layer_feedForw_full(Layer%d):\n", self->lid);
    fprintf(stderr, "  outputs = [");
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", self->outputs[i]);
    }
    fprintf(stderr, "]\n  gradients = [");
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", self->gradients[i]);
    }
    fprintf(stderr, "]\n");
#endif
}

static void Layer_feedBack_full(Layer* self)
{
    assert (self->ltype == LAYER_FULL);
    assert (self->lprev != NULL);
    Layer* lprev = self->lprev;

    /* Clear errors. */
    for (int j = 0; j < lprev->nnodes; j++) {
        lprev->errors[j] = 0;
    }

    int k = 0;
    for (int i = 0; i < self->nnodes; i++) {
        /* Computer the weight/bias updates. */
        double dnet = self->errors[i] * self->gradients[i];
        for (int j = 0; j < lprev->nnodes; j++) {
            /* Propagate the errors to the previous layer. */
            lprev->errors[j] += self->weights[k] * dnet;
            self->u_weights[k] += dnet * lprev->outputs[j];
            k++;
        }
        self->u_biases[i] += dnet;
    }

#if DEBUG_LAYER
    fprintf(stderr, "Layer_feedBack_full(Layer%d):\n", self->lid);
    for (int i = 0; i < self->nnodes; i++) {
        double dnet = self->errors[i] * self->gradients[i];
        fprintf(stderr, "  dnet = %.4f, dw = [", dnet);
        for (int j = 0; j < lprev->nnodes; j++) {
            double dw = dnet * lprev->outputs[j];
            fprintf(stderr, " %.4f", dw);
        }
        fprintf(stderr, "]\n");
    }
#endif
}

/* Layer_feedForw_conv(self)
   Performs feed forward updates.
*/
static void Layer_feedForw_conv(Layer* self)
{
    assert (self->ltype == LAYER_CONV);
    assert (self->lprev != NULL);
    Layer* lprev = self->lprev;

    int kernsize = self->conv.kernsize;
    int i = 0;
    for (int z1 = 0; z1 < self->depth; z1++) {
        /* z1: dst matrix */
        /* qbase: kernel matrix base index */
        int qbase = z1 * lprev->depth * kernsize * kernsize;
        for (int y1 = 0; y1 < self->height; y1++) {
            int y0 = self->conv.stride * y1 - self->conv.padding;
            for (int x1 = 0; x1 < self->width; x1++) {
                int x0 = self->conv.stride * x1 - self->conv.padding;
                /* Compute the kernel at (x1,y1) */
                /* (x0,y0): src pixel */
                double v = self->biases[z1];
                for (int z0 = 0; z0 < lprev->depth; z0++) {
                    /* z0: src matrix */
                    /* pbase: src matrix base index */
                    int pbase = z0 * lprev->width * lprev->height;
                    for (int dy = 0; dy < kernsize; dy++) {
                        int y = y0+dy;
                        if (0 <= y && y < lprev->height) {
                            int p = pbase + y*lprev->width;
                            int q = qbase + dy*kernsize;
                            for (int dx = 0; dx < kernsize; dx++) {
                                int x = x0+dx;
                                if (0 <= x && x < lprev->width) {
                                    v += lprev->outputs[p+x] * self->weights[q+dx];
                                }
                            }
                        }
                    }
                }
                /* Apply the activation function. */
                v = relu(v);
                self->outputs[i] = v;
                self->gradients[i] = relu_g(v);
                i++;
            }
        }
    }
    assert (i == self->nnodes);

#if DEBUG_LAYER
    fprintf(stderr, "Layer_feedForw_conv(Layer%d):\n", self->lid);
    fprintf(stderr, "  outputs = [");
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", self->outputs[i]);
    }
    fprintf(stderr, "]\n  gradients = [");
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", self->gradients[i]);
    }
    fprintf(stderr, "]\n");
#endif
}

static void Layer_feedBack_conv(Layer* self)
{
    assert (self->ltype == LAYER_CONV);
    assert (self->lprev != NULL);
    Layer* lprev = self->lprev;

    /* Clear errors. */
    for (int j = 0; j < lprev->nnodes; j++) {
        lprev->errors[j] = 0;
    }

    int kernsize = self->conv.kernsize;
    int i = 0;
    for (int z1 = 0; z1 < self->depth; z1++) {
        /* z1: dst matrix */
        /* qbase: kernel matrix base index */
        int qbase = z1 * lprev->depth * kernsize * kernsize;
        for (int y1 = 0; y1 < self->height; y1++) {
            int y0 = self->conv.stride * y1 - self->conv.padding;
            for (int x1 = 0; x1 < self->width; x1++) {
                int x0 = self->conv.stride * x1 - self->conv.padding;
                /* Compute the kernel at (x1,y1) */
                /* (x0,y0): src pixel */
                double dnet = self->errors[i] * self->gradients[i];
                for (int z0 = 0; z0 < lprev->depth; z0++) {
                    /* z0: src matrix */
                    /* pbase: src matrix base index */
                    int pbase = z0 * lprev->width * lprev->height;
                    for (int dy = 0; dy < kernsize; dy++) {
                        int y = y0+dy;
                        if (0 <= y && y < lprev->height) {
                            int p = pbase + y*lprev->width;
                            int q = qbase + dy*kernsize;
                            for (int dx = 0; dx < kernsize; dx++) {
                                int x = x0+dx;
                                if (0 <= x && x < lprev->width) {
                                    lprev->errors[p+x] += self->weights[q+dx] * dnet;
                                    self->u_weights[q+dx] += dnet * lprev->outputs[p+x];
                                }
                            }
                        }
                    }
                }
                self->u_biases[z1] += dnet;
                i++;
            }
        }
    }
    assert (i == self->nnodes);

#if DEBUG_LAYER
    fprintf(stderr, "Layer_feedBack_conv(Layer%d):\n", self->lid);
    for (int i = 0; i < self->nnodes; i++) {
        double dnet = self->errors[i] * self->gradients[i];
        fprintf(stderr, "  dnet=%.4f, dw=[", dnet);
        for (int j = 0; j < lprev->nnodes; j++) {
            double dw = dnet * lprev->outputs[j];
            fprintf(stderr, " %.4f", dw);
        }
        fprintf(stderr, "]\n");
    }
#endif
}

/* Layer_setInputs(self, values)
   Sets the input values.
*/
void Layer_setInputs(Layer* self, const double* values)
{
    assert (self != NULL);
    assert (self->ltype == LAYER_INPUT);
    assert (self->lprev == NULL);

#if DEBUG_LAYER
    fprintf(stderr, "Layer_setInputs(Layer%d): values = [", self->lid);
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", values[i]);
    }
    fprintf(stderr, "]\n");
#endif

    /* Set the values as the outputs. */
    for (int i = 0; i < self->nnodes; i++) {
        self->outputs[i] = values[i];
    }

    /* Start feed forwarding. */
    Layer* layer = self->lnext;
    while (layer != NULL) {
        switch (layer->ltype) {
        case LAYER_FULL:
            Layer_feedForw_full(layer);
            break;
        case LAYER_CONV:
            Layer_feedForw_conv(layer);
            break;
        default:
            break;
        }
        layer = layer->lnext;
    }
}

/* Layer_getOutputs(self, outputs)
   Gets the output values.
*/
void Layer_getOutputs(const Layer* self, double* outputs)
{
    assert (self != NULL);
    for (int i = 0; i < self->nnodes; i++) {
        outputs[i] = self->outputs[i];
    }
}

/* Layer_getErrorTotal(self)
   Gets the error total.
*/
double Layer_getErrorTotal(const Layer* self)
{
    assert (self != NULL);
    double total = 0;
    for (int i = 0; i < self->nnodes; i++) {
        double e = self->errors[i];
        total += e*e;
    }
    return (total / self->nnodes);
}

/* Layer_learnOutputs(self, values)
   Learns the output values.
*/
void Layer_learnOutputs(Layer* self, const double* values)
{
    assert (self != NULL);
    assert (self->ltype != LAYER_INPUT);
    assert (self->lprev != NULL);
    for (int i = 0; i < self->nnodes; i++) {
        self->errors[i] = (self->outputs[i] - values[i]);
    }

#if DEBUG_LAYER
    fprintf(stderr, "Layer_learnOutputs(Layer%d): errors = [", self->lid);
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", self->errors[i]);
    }
    fprintf(stderr, "]\n");
#endif

    /* Start backpropagation. */
    Layer* layer = self;
    while (layer != NULL) {
        switch (layer->ltype) {
        case LAYER_FULL:
            Layer_feedBack_full(layer);
            break;
        case LAYER_CONV:
            Layer_feedBack_conv(layer);
            break;
        default:
            break;
        }
        layer = layer->lprev;
    }
}

/* Layer_update(self, rate)
   Updates the weights.
*/
void Layer_update(Layer* self, double rate)
{
    for (int i = 0; i < self->nbiases; i++) {
        self->biases[i] -= rate * self->u_biases[i];
        self->u_biases[i] = 0;
    }
    for (int i = 0; i < self->nweights; i++) {
        self->weights[i] -= rate * self->u_weights[i];
        self->u_weights[i] = 0;
    }
    if (self->lprev != NULL) {
        Layer_update(self->lprev, rate);
    }
}

/* Layer_create_input(depth, width, height)
   Creates an input Layer with size (depth x weight x height).
*/
Layer* Layer_create_input(int depth, int width, int height)
{
    return Layer_create(
        NULL, LAYER_INPUT, depth, width, height, 0, 0);
}

/* Layer_create_full(lprev, nnodes, std)
   Creates a fully-connected Layer.
*/
Layer* Layer_create_full(Layer* lprev, int nnodes, double std)
{
    assert (lprev != NULL);
    Layer* self = Layer_create(
        lprev, LAYER_FULL, nnodes, 1, 1,
        nnodes, nnodes * lprev->nnodes);
    assert (self != NULL);

    for (int i = 0; i < self->nweights; i++) {
        self->weights[i] = std * nrnd();
    }

#if DEBUG_LAYER
    Layer_dump(self, stderr);
#endif
    return self;
}

/* Layer_create_conv(lprev, depth, width, height, kernsize, padding, stride, std)
   Creates a convolutional Layer.
*/
Layer* Layer_create_conv(
    Layer* lprev, int depth, int width, int height,
    int kernsize, int padding, int stride, double std)
{
    assert (lprev != NULL);
    assert ((kernsize % 2) == 1);
    assert ((width-1) * stride + kernsize <= lprev->width + padding*2);
    assert ((height-1) * stride + kernsize <= lprev->height + padding*2);

    Layer* self = Layer_create(
        lprev, LAYER_CONV, depth, width, height,
        depth, depth * lprev->depth * kernsize * kernsize);
    assert (self != NULL);

    self->conv.kernsize = kernsize;
    self->conv.padding = padding;
    self->conv.stride = stride;

    for (int i = 0; i < self->nweights; i++) {
        self->weights[i] = std * nrnd();
    }

#if DEBUG_LAYER
    Layer_dump(self, stderr);
#endif
    return self;
}




/****************************************************************************BOOM*******************************************************************************************/

/*  IdxFile
 */
typedef struct _IdxFile
{
    int ndims;
    uint32_t* dims;
    uint8_t* data;
} IdxFile;

#define DEBUG_IDXFILE 0

/* IdxFile_read(fp)
   Reads all the data from given fp.
*/

IdxFile* IdxFile_read(FILE* fp)
{
    /* Read the file header. */
    struct {
        uint8_t magic[2];
        uint8_t type;
        uint8_t ndims;
    } header;
    if (fread(&header, sizeof(header), 1, fp) != 1) return NULL;

    if (header.magic[0] != 0 || header.magic[1] != 0) return NULL;
    if (header.type != 0x08) return NULL;
    if (header.ndims < 1) return NULL;

    /* Read the dimensions. */
    IdxFile* self = (IdxFile*)calloc(1, sizeof(IdxFile));
    if (self == NULL) return NULL;
    self->ndims = header.ndims;
    self->dims = (uint32_t*)calloc(self->ndims, sizeof(uint32_t));
    if (self->dims == NULL) return NULL;

    if (fread(self->dims, sizeof(uint32_t), self->ndims, fp) == self->ndims) {
        uint32_t nbytes = sizeof(uint8_t);
        for (int i = 0; i < self->ndims; i++) {
            /* Fix the byte order manually. */
            uint32_t size = (self->dims[i] << 24) |
                            ((self->dims[i] << 8) & 0x00FF0000) |
                            ((self->dims[i] >> 8) & 0x0000FF00) |
                            (self->dims[i] >> 24);
            nbytes *= size;
            self->dims[i] = size;
        }
        /* Read the data. */
        self->data = (uint8_t*) malloc(nbytes);
        if (self->data != NULL) {
            fread(self->data, sizeof(uint8_t), nbytes, fp);
        }
    }

    return self;
}


// IdxFile* IdxFile_read(FILE* fp)
// {
//     /* Read the file header. */
//     struct {
//         uint16_t magic;
//         uint8_t type;
//         uint8_t ndims;
//         /* big endian */
//     } header;
//     if (fread(&header, sizeof(header), 1, fp) != 1) return NULL;
// #if DEBUG_IDXFILE
//     fprintf(stderr, "IdxFile_read: magic=%x, type=%x, ndims=%u\n",
//             header.magic, header.type, header.ndims);
// #endif
//     if (header.magic != 0) return NULL;
//     if (header.type != 0x08) return NULL;
//     if (header.ndims < 1) return NULL;

//     /* Read the dimensions. */
//     IdxFile* self = (IdxFile*)calloc(1, sizeof(IdxFile));
//     if (self == NULL) return NULL;
//     self->ndims = header.ndims;
//     self->dims = (uint32_t*)calloc(self->ndims, sizeof(uint32_t));
//     if (self->dims == NULL) return NULL;
    
//     if (fread(self->dims, sizeof(uint32_t), self->ndims, fp) == self->ndims) {
//         uint32_t nbytes = sizeof(uint8_t);
//         for (int i = 0; i < self->ndims; i++) {
//             /* Fix the byte order. */
//             uint32_t size = be32toh(self->dims[i]);
// #if DEBUG_IDXFILE
//             fprintf(stderr, "IdxFile_read: size[%d]=%u\n", i, size);
// #endif
//             nbytes *= size;
//             self->dims[i] = size;
//         }
//         /* Read the data. */
//         self->data = (uint8_t*) malloc(nbytes);
//         if (self->data != NULL) {
//             fread(self->data, sizeof(uint8_t), nbytes, fp);
// #if DEBUG_IDXFILE
//             fprintf(stderr, "IdxFile_read: read: %lu bytes\n", n);
// #endif
//         }
//     }

//     return self;
// }



/* IdxFile_destroy(self)
   Release the memory.
*/
void IdxFile_destroy(IdxFile* self)
{
    assert (self != NULL);
    if (self->dims != NULL) {
        free(self->dims);
        self->dims = NULL;
    }
    if (self->data != NULL) {
        free(self->data);
        self->data = NULL;
    }
    free(self);
}

/* IdxFile_get1(self, i)
   Get the i-th record of the Idx1 file. (uint8_t)
 */
uint8_t IdxFile_get1(IdxFile* self, int i)
{
    assert (self != NULL);
    assert (self->ndims == 1);
    assert (i < self->dims[0]);
    return self->data[i];
}

/* IdxFile_get3(self, i, out)
   Get the i-th record of the Idx3 file. (matrix of uint8_t)
 */
void IdxFile_get3(IdxFile* self, int i, uint8_t* out)
{
    assert (self != NULL);
    assert (self->ndims == 3);
    assert (i < self->dims[0]);
    size_t n = self->dims[1] * self->dims[2];
    memcpy(out, &self->data[i*n], n);
}


/* main */
int main(int argc, char* argv[])
{
    
    /* argv[1] = train images */
    /* argv[2] = train labels */
    /* argv[3] = test images */
    /* argv[4] = test labels */
    if (argc < 4) return 100;
    

    /* Use a fixed random seed for debugging. */
    srand(0);
    /* Initialize layers. */
    /* Input layer - 1x28x28. */
    Layer* linput = Layer_create_input(1, 28, 28);
    /* Conv1 layer - 16x14x14, 3x3 conv, padding=1, stride=2. */
    /* (14-1)*2+3 < 28+1*2 */
    Layer* lconv1 = Layer_create_conv(linput, 16, 14, 14, 3, 1, 2, 0.1);
    /* Conv2 layer - 32x7x7, 3x3 conv, padding=1, stride=2. */
    /* (7-1)*2+3 < 14+1*2 */
    Layer* lconv2 = Layer_create_conv(lconv1, 32, 7, 7, 3, 1, 2, 0.1);
    /* FC1 layer - 200 nodes. */
    Layer* lfull1 = Layer_create_full(lconv2, 200, 0.1);
    /* FC2 layer - 200 nodes. */
    Layer* lfull2 = Layer_create_full(lfull1, 200, 0.1);
    /* Output layer - 10 nodes. */
    Layer* loutput = Layer_create_full(lfull2, 10, 0.1);

    /* Read the training images & labels. */
    IdxFile* images_train = NULL;
    {
        FILE* fp = fopen(argv[1], "rb");
        if (fp == NULL) return 111;
        images_train = IdxFile_read(fp);
        if (images_train == NULL) return 111;
        fclose(fp);
    }
    IdxFile* labels_train = NULL;
    {
        FILE* fp = fopen(argv[2], "rb");
        if (fp == NULL) return 111;
        labels_train = IdxFile_read(fp);
        if (labels_train == NULL) return 111;
        fclose(fp);
    }

    fprintf(stderr, "training...\n");
    double rate = 0.1;
    double etotal = 0;
    int nepoch = 1;
    int batch_size = 32;
    int train_size = images_train->dims[0];
    for (int i = 0; i < nepoch * train_size; i++) {
        /* Pick a random sample from the training data */
        uint8_t img[28*28];
        double x[28*28];
        double y[10];
        int index = rand() % train_size;
        IdxFile_get3(images_train, index, img);
        for (int j = 0; j < 28*28; j++) {
            x[j] = img[j]/255.0;
        }
        Layer_setInputs(linput, x);
        Layer_getOutputs(loutput, y);
        int label = IdxFile_get1(labels_train, index);
#if 0
        fprintf(stderr, "label=%u, y=[", label);
        for (int j = 0; j < 10; j++) {
            fprintf(stderr, " %.3f", y[j]);
        }
        fprintf(stderr, "]\n");
#endif
        for (int j = 0; j < 10; j++) {
            y[j] = (j == label)? 1 : 0;
        }
        Layer_learnOutputs(loutput, y);
        etotal += Layer_getErrorTotal(loutput);
        if ((i % batch_size) == 0) {
            /* Minibatch: update the network for every n samples. */
            Layer_update(loutput, rate/batch_size);
        }
        if ((i % 1000) == 0) {
            fprintf(stderr, "i=%d, error=%.4f\n", i, etotal/1000);
            etotal = 0;
        }
    }

    IdxFile_destroy(images_train);
    IdxFile_destroy(labels_train);

    /* Training finished. */

    //Layer_dump(linput, stdout);
    //Layer_dump(lconv1, stdout);
    //Layer_dump(lconv2, stdout);
    //Layer_dump(lfull1, stdout);
    //Layer_dump(lfull2, stdout);
    //Layer_dump(loutput, stdout);

    /* Read the test images & labels. */
    
    IdxFile* images_test = NULL;
    {
        FILE* fp = fopen(argv[3], "rb");
        if (fp == NULL) return 111;
        images_test = IdxFile_read(fp);
        if (images_test == NULL) return 111;
        fclose(fp);
    }
    IdxFile* labels_test = NULL;
    {
        FILE* fp = fopen(argv[4], "rb");
        if (fp == NULL) return 111;
        labels_test = IdxFile_read(fp);
        if (labels_test == NULL) return 111;
        fclose(fp);
    }

    fprintf(stderr, "testing...\n");
    int ntests = images_test->dims[0];
    int ncorrect = 0;
    for (int i = 0; i < ntests; i++) {
        uint8_t img[28*28];
        double x[28*28];
        double y[10];
        IdxFile_get3(images_test, i, img);
        for (int j = 0; j < 28*28; j++) {
            x[j] = img[j]/255.0;
        }
        Layer_setInputs(linput, x);
        Layer_getOutputs(loutput, y);
        int label = IdxFile_get1(labels_test, i);
        /* Pick the most probable label. */
        int mj = -1;
        for (int j = 0; j < 10; j++) {
            if (mj < 0 || y[mj] < y[j]) {
                mj = j;
            }
        }
        if (mj == label) {
            ncorrect++;
        }
        if ((i % 1000) == 0) {
            fprintf(stderr, "i=%d\n", i);
        }
    }
    fprintf(stderr, "ntests=%d, ncorrect=%d\n", ntests, ncorrect);

    IdxFile_destroy(images_test);
    IdxFile_destroy(labels_test);

    Layer_destroy(linput);
    Layer_destroy(lconv1);
    Layer_destroy(lconv2);
    Layer_destroy(lfull1);
    Layer_destroy(lfull2);
    Layer_destroy(loutput);

    return 0;
}
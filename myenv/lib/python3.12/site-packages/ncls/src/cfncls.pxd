from libc.stdint cimport int64_t


cdef extern from "stdlib.h":
    void free(void *)
    void *malloc(size_t)
    void *calloc(size_t,size_t)
    void *realloc(void *,size_t)
    int c_abs "abs" (int)
    void qsort(void *base, size_t nmemb, size_t size,
            int (*compar)(void *,void *))

cdef extern from "stdio.h":
    ctypedef struct FILE:
        pass

    FILE *fopen(char *,char *)
    FILE *open_memstream(void *, size_t *)
    FILE *fmemopen (void *, size_t, const char *)

    int fclose(FILE *)
    int fflush(FILE *)
    int sscanf(char *str,char *fmt,...)
    int sprintf(char *str,char *fmt,...)
    int fprintf(FILE *ifile,char *fmt,...)
    char *fgets(char *str,int size,FILE *ifile)


cdef extern from "ncls/src/fintervaldb.h":
    ctypedef struct IntervalMap:
        double start
        double end
        int target_id
        int sublist

    ctypedef struct IntervalIterator:
        pass

    ctypedef struct SublistHeader:
        int start
        int len

    int find_overlap_start(double start, double end, IntervalMap im[], int n)
    int imstart_qsort_cmp(void *void_a,void *void_b)
    # int target_qsort_cmp(void *void_a,void *void_b)
    IntervalMap *read_intervals(int n,FILE *ifile)
    SublistHeader *build_nested_list(IntervalMap im[],int n,int *p_n,int *p_nlists)
    SublistHeader *build_nested_list_inplace(IntervalMap im[],int n,int *p_n,int *p_nlists)
    IntervalMap *interval_map_alloc(int n)
    IntervalIterator *interval_iterator_alloc()
    int free_interval_iterator(IntervalIterator *it)
    IntervalIterator *reset_interval_iterator(IntervalIterator *it)
    int *alloc_array(int n)
    int find_intervals_stack(int start_stack[], int end_stack[], int sp, int start,
                             int end, IntervalMap im[], int n,
                             SublistHeader subheader[], IntervalMap buf[],
                             int *nfound)

    int find_intervals(IntervalIterator *it0,
                       double start,
                       double end,
                       IntervalMap im[],
                       int n,
                       SublistHeader subheader[],
                       int nlists,
                       IntervalMap buf[],
                       int nbuf,
                       int *p_nreturn,
                       IntervalIterator **it_return)
    void find_k_next(int start, int end,
                    IntervalMap im[], int n,
                    SublistHeader subheader[], int nlists,
                    IntervalMap buf[], int ktofind,
                    int *p_nreturn)

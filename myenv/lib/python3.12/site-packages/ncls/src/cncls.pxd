from libc.stdint cimport int64_t

# cdef extern from "string.h":
#   ctypedef int size_t
#   void *memcpy(void *dst,void *src,size_t len)
#   void *memmove(void *dst,void *src,size_t len)
#   void *memset(void *b,int c,size_t len)

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

# cdef extern from "string.h":
#   int strcmp(char *s1, char *s2)
#   int strncmp(char *s1,char *s2,size_t len)
#   char *strcpy(char *dest,char *src)
#   char *strdup(char *)
#   char *strcat(char *,char *)


cdef extern from "ncls/src/intervaldb.h":
    ctypedef struct IntervalMap:
        int64_t start
        int64_t end
        int target_id
        int sublist

    ctypedef struct IntervalIterator:
        pass

    ctypedef struct SublistHeader:
        int start
        int len

    int64_t find_overlap_start(int64_t start, int64_t end, IntervalMap im[], int n)
    int64_t find_suboverlap_start(int64_t start, int64_t end, int isub, IntervalMap im[], SublistHeader subheader[])
    int imstart_qsort_cmp(void *void_a,void *void_b)
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
                       int64_t start,
                       int64_t end,
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

    # char *write_binary_files(IntervalMap im[],int n,int ntop,int div,SublistHeader *subheader,int nlists,char filestem[])
    # IntervalDBFile *read_binary_files(char filestem[],char err_msg[],int subheader_nblock) except NULL
    # int free_interval_dbfile(IntervalDBFile *db_file)
    # int find_file_intervals(IntervalIterator *it0,int start,int end,IntervalIndex ii[],int nii,SublistHeader subheader[],int nlists,SubheaderFile *subheader_file,int ntop,int div,FILE *ifile,IntervalMap buf[],int nbuf,int *p_nreturn,IntervalIterator **it_return) except -1
    # int write_padded_binary(IntervalMap im[],int n,int div,FILE *ifile)

    int read_imdiv(FILE *ifile,IntervalMap imdiv[],int div,int i_div,int ntop)

    # int save_text_file(char filestem[],char basestem[],char err_msg[],FILE *ofile)
    # int text_file_to_binaries(FILE *infile,char buildpath[],char err_msg[])
    # int C_int_max

# cdef extern from "ncls/src/utarray.h":

#     ctypedef struct UT_icd:
#         pass

#     ctypedef struct UT_array:
#         pass

#     const UT_icd ut_int_icd

#     utarray_free(UT_array *a)
#     UT_array utarray_new(UT_array *a, UT_icd *icd)
#     utarray_len(UT_array *a)
#     int* utarray_eltptr(UT_array *a, int j)
#     utarray_push_back(UT_array *a, void *p)


# cdef class NCLS:
#     cdef int n
#     cdef int ntop
#     cdef int nlists
#     cdef IntervalMap *im
#     cdef SublistHeader *subheader


# cdef extern from "ncls/src/utarray.h":

#     ctypedef struct UT_icd:
#         pass

#     ctypedef struct UT_array:
#         pass

#     const UT_icd ut_int_icd

#     void utarray_new(UT_array *a, UT_icd *icd)
#     int utarray_len(UT_array *a)
#     int* utarray_eltptr(UT_array *a, int j)
#     void utarray_push_back(UT_array *a, void *p)
#     void utarray_free(UT_array *a)

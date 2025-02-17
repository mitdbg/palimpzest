
#ifndef INTERVALDB_HEADER_INCLUDED
#define INTERVALDB_HEADER_INCLUDED 1
#include "default.h"
#include <limits.h>

#include <stdint.h>

extern int C_int_max;

typedef struct {
  int64_t start;
  int64_t end;
  int64_t target_id;
  /* int target_start; */
  /* int target_end; */
  int64_t sublist;
} IntervalMap;


typedef struct {
  int64_t start;
  int64_t end;
} IntervalIndex;

typedef struct {
  int start;
  int len;
} SublistHeader;

typedef struct {
  int n;
  int ntop;
  int nlists;
  IntervalMap *im;
  SublistHeader *subheader;
} IntervalDB;

typedef struct { /* FOR REAL-TIME DISK ACCESS TO SUBLIST HEADER FILE*/
  SublistHeader *subheader;
  int nblock;
  int start;
  FILE *ifile;
} SubheaderFile;

typedef struct {
  int n;
  int ntop;
  int nlists;
  int div;
  int nii;
  IntervalIndex *ii;
  SublistHeader *subheader;
  SubheaderFile subheader_file;
  FILE *ifile_idb;
} IntervalDBFile;

typedef struct IntervalIterator_S {
  int i;
  int n;
  int nii;
  int ntop;
  int i_div;
  IntervalMap *im;
  struct IntervalIterator_S *up;
  struct IntervalIterator_S *down;
} IntervalIterator;


typedef struct {
  FILE *ifile;
  int left;
  int right;
  int ihead;
  char *filename;
} FilePtrRecord;

extern int *alloc_array(int n);

extern int64_t find_overlap_start(int64_t start, int64_t end, IntervalMap im[],int n);
extern int64_t find_suboverlap_start(int64_t start, int64_t end,int isub, IntervalMap im[], SublistHeader subheader[]);
extern int imstart_qsort_cmp(const void *void_a,const void *void_b);
extern int target_qsort_cmp(const void *void_a,const void *void_b);
extern IntervalMap *read_intervals(int n,FILE *ifile);
extern SublistHeader *build_nested_list(IntervalMap im[],int n,
					int *p_n,int *p_nlists);
extern SublistHeader *build_nested_list_inplace(IntervalMap im[],int n,
                                                int *p_n,int *p_nlists);
extern IntervalMap *interval_map_alloc(int n);
extern IntervalDB *build_interval_db(IntervalMap im[],int n);
extern IntervalIterator *interval_iterator_alloc(void);
extern int free_interval_iterator(IntervalIterator *it);
extern IntervalIterator *reset_interval_iterator(IntervalIterator *it);
extern int64_t find_intervals(IntervalIterator *it0,int64_t start,int64_t end,IntervalMap im[],int n,SublistHeader subheader[],int nlists,IntervalMap buf[],int nbuf,int *p_nreturn,IntervalIterator **it_return);
extern int read_imdiv(FILE *ifile,IntervalMap imdiv[],int div,int i_div,int ntop);
extern IntervalMap *read_sublist(FILE *ifile,SublistHeader *subheader,IntervalMap *im);
extern int find_file_intervals(IntervalIterator *it0,int start,int end,
			       IntervalIndex ii[],int nii,
			       SublistHeader subheader[],int nlists,
			       SubheaderFile *subheader_file,
			       int ntop,int div,FILE *ifile,
			       IntervalMap buf[],int nbuf,
			       int *p_nreturn,IntervalIterator **it_return);
extern int write_padded_binary(IntervalMap im[],int n,int div,FILE *ifile);
extern char *write_binary_files(IntervalMap im[],int n,int ntop,int div,
				SublistHeader *subheader,int nlists,char filestem[]);
extern IntervalDBFile *read_binary_files(char filestem[],char err_msg[],
					 int subheader_nblock);
extern int free_interval_dbfile(IntervalDBFile *db_file);

extern int save_text_file(char filestem[],char err_msg[],
		  char basestem[],FILE *ofile);
extern int text_file_to_binaries(FILE *infile,char buildpath[],char err_msg[]);
extern void reorient_intervals(int n,IntervalMap im[],int ori_sign);

/* extern int find_k_next(int start, int end, */
/*                 IntervalMap im[], int n, */
/*                 SublistHeader subheader[], int nlists, */
/*                 IntervalMap buf[], int ktofind, */
/*                 int *p_nreturn); */

#define FIND_FILE_MALLOC_ERR -2

#define ITERATOR_STACK_TOP(it) while (it->up) it=it->up;
#define FREE_ITERATOR_STACK(it,it2,it_next) \
  for (it2=it->down;it2;it2=it_next) { \
    it_next=it2->down; \
    if (it2->im) \
      free(it2->im); \
    free(it2); \
  } \
  for (it2=it;it2;it2=it_next) { \
    it_next=it2->up; \
    if (it2->im) \
      free(it2->im); \
    free(it2); \
  }

#define PUSH_ITERATOR_STACK(it,it2,TYPE) \
  if (it->down) \
    it2=it->down; \
  else { \
    CALLOC(it2,1,TYPE); \
    it2->up = it; \
    it->down= it2; \
  }

/* IF it->up NON-NULL, MOVE UP, EVAL FALSE.
   IF NULL, DON'T ASSIGN it, BUT EVAL TRUE */
#define POP_ITERATOR_STACK_DONE(it) (it->up==NULL || (it=it->up)==NULL)

#define POP_ITERATOR_STACK(it) (it->up && (it=it->up))

/* #define MALLOC_INT_ARRAY(n) ((int*) malloc (n)) */


#ifdef MERGE_INTERVAL_ORIENTATIONS
/* MACROS FOR MERGING POSITIVE AND NEGATIVE ORIENTATIONS */
#define START_POSITIVE(IM) (((IM).start>=0) ? ((IM).start) : -((IM).end))
#define END_POSITIVE(IM) (((IM).start>=0) ? ((IM).end) : -((IM).start))
#define SET_INTERVAL_POSITIVE(IM,START,END) if ((IM).start>=0) {\
  START= (IM).start; \
  END=   (IM).end; \
} else { \
  START= -((IM).end); \
  END=   -((IM).start); \
}

#define HAS_OVERLAP_POSITIVE(IM,START,END) (((IM).start>=0) ? \
    ((IM).start<(END) && (START)<(IM).end) \
  : (-((IM).end)<(END) && (START) < -((IM).start)))
 /* ????? MERGE_INTERVAL_ORIENTATIONS ??????? */

#else
/* STANDARD MACROS */
#define START_POSITIVE(IM) ((IM).start)
#define END_POSITIVE(IM) ((IM).end)
#define HAS_OVERLAP_POSITIVE(IM,START,END) ((IM).start<(END) && (START)<(IM).end)

#endif

/* STORE ALL INTERVALS IN POSITIVE SOURCE ORIENTATION */
#define ALL_POSITIVE_ORIENTATION 1
/* ONLY LOAD SUBLISTS INDIVIDUALLY WHEN NEEDED */
#define ON_DEMAND_SUBLIST_HEADER 1



#endif

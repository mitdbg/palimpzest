

#include "intervaldb32.h"


int imstart_qsort_cmp(const void *void_a,const void *void_b)
{ /* STRAIGHTFORWARD COMPARISON OF SIGNED start VALUES, LONGER INTERVALS 1ST */
  IntervalMap *a=(IntervalMap *)void_a,*b=(IntervalMap *)void_b;
  if (a->start<b->start)
    return -1;
  else if (a->start>b->start)
    return 1;
  else if (a->end>b->end) /* SAME START: PUT LONGER INTERVAL 1ST */
    return -1;
  else if (a->end<b->end) /* CONTAINED INTERVAL SHOULD FOLLOW LARGER INTERVAL*/
    return 1;
  else
    return 0;
}




#ifdef MERGE_INTERVAL_ORIENTATIONS
int im_qsort_cmp(const void *void_a,const void *void_b)
{ /* MERGE FORWARD AND REVERSE INTERVALS AS IF THEY WERE ALL IN FORWARD ORI */
  int a_start,a_end,b_start,b_end;
  IntervalMap *a=(IntervalMap *)void_a,*b=(IntervalMap *)void_b;
  SET_INTERVAL_POSITIVE(*a,a_start,a_end);
  SET_INTERVAL_POSITIVE(*b,b_start,b_end);
  if (a_start<b_start)
    return -1;
  else if (a_start>b_start)
    return 1;
  else if (a_end>b_end) /* SAME START: PUT LONGER INTERVAL 1ST */
    return -1;
  else if (a_end<b_end) /* CONTAINED INTERVAL SHOULD FOLLOW LARGER INTERVAL*/
    return 1;
  else
    return 0;
}
#endif

int sublist_qsort_cmp(const void *void_a,const void *void_b)
{ /* SORT IN SUBLIST ORDER, SECONDARILY BY start */
  IntervalMap *a=(IntervalMap *)void_a,*b=(IntervalMap *)void_b;
  if (a->sublist<b->sublist)
    return -1;
  else if (a->sublist>b->sublist)
    return 1;
  else if (START_POSITIVE(*a) < START_POSITIVE(*b))
    return -1;
  else if (START_POSITIVE(*a) > START_POSITIVE(*b))
    return 1;
  else
    return 0;
}


SublistHeader *build_nested_list(IntervalMap im[],int n,
                                 int *p_n,int *p_nlists)
{
  int i=0,j,k,parent,nsub=0,nlists=0;
  IntervalMap *imsub=NULL;
  SublistHeader *subheader=NULL;

/* #ifdef ALL_POSITIVE_ORIENTATION */
/*   reorient_intervals(n,im,1); /\* FORCE ALL INTERVALS INTO POSITIVE ORI *\/ */
/* #endif */
#ifdef MERGE_INTERVAL_ORIENTATIONS
  qsort(im,n,sizeof(IntervalMap),im_qsort_cmp); /* SORT BY start, CONTAINMENT */
#else
  qsort(im,n,sizeof(IntervalMap),imstart_qsort_cmp); /* SORT BY start, CONTAINMENT */
#endif
  while (i<n) { /* TOP LEVEL LIST SCAN */
    parent=i;
    i=parent+1;
    while (i<n && parent>=0) { /* RECURSIVE ALGORITHM OF ALEX ALEKSEYENKO */
      if (END_POSITIVE(im[i])>END_POSITIVE(im[parent]) /* i NOT CONTAINED */
          || (END_POSITIVE(im[i])==END_POSITIVE(im[parent]) /* SAME INTERVAL! */
              && START_POSITIVE(im[i])==START_POSITIVE(im[parent])))
        parent=im[parent].sublist; /* POP RECURSIVE STACK*/
      else  { /* i CONTAINED IN parent*/
        im[i].sublist=parent; /* MARK AS CONTAINED IN parent */
        nsub++; /* COUNT TOTAL #SUBLIST ENTRIES */
        parent=i; /* AND PUSH ONTO RECURSIVE STACK */
        i++; /* ADVANCE TO NEXT INTERVAL */
      }
    }
  } /* AT THIS POINT sublist IS EITHER -1 IF NOT IN SUBLIST, OR INDICATES parent*/

  if (nsub>0) { /* WE HAVE SUBLISTS TO PROCESS */
    CALLOC(imsub,nsub,IntervalMap); /* TEMPORARY ARRAY FOR REPACKING SUBLISTS */
    for (i=j=0;i<n;i++) { /* GENERATE LIST FOR SORTING; ASSIGN HEADER INDEXES*/
      parent=im[i].sublist;
      /* printf("Interval %i has parent %d\n", i, parent); */
      if (parent>=0)  {/* IN A SUBLIST */
        imsub[j].start=i;
        imsub[j].sublist=parent;
        j++;
        if (im[parent].sublist<0){ /* A NEW PARENT! SET HIS SUBLIST HEADER INDEX */
          /* printf("Setting parent %d to sublist %d\n", parent, nlists += 1); */
          im[parent].sublist=nlists++;
        }
      }
      im[i].sublist= -1; /* RESET TO DEFAULT VALUE: NO SUBLIST */
    }
    qsort(imsub,nsub,sizeof(IntervalMap),sublist_qsort_cmp);
    /* AT THIS POINT SUBLISTS ARE GROUPED TOGETHER, READY TO PACK */

    CALLOC(subheader,nlists,SublistHeader); /* SUBLIST HEADER INDEX */
    for (i=0;i<nsub;i++) { /* COPY SUBLIST ENTRIES TO imsub */
      j=imsub[i].start;
      /* printf("j: %d\n", j); */
      parent=imsub[i].sublist;
      memcpy(imsub+i,im+j,sizeof(IntervalMap)); /* COPY INTERVAL */
      k=im[parent].sublist;
      /* printf("k is %d\n", k); */
      if (subheader[k].len==0) /* START A NEW SUBLIST */
        subheader[k].start=i;
      subheader[k].len++; /* COUNT THE SUBLIST ENTRIES */
      im[j].start=im[j].end= -1; /* MARK FOR DELETION */
    } /* DONE COPYING ALL SUBLISTS TO imsub */

    for (i=j=0;i<n;i++) /* COMPRESS THE LIST TO REMOVE SUBLISTS */
      if (im[i].start!= -1 || im[i].end!= -1) { /* NOT IN A SUBLIST, SO KEEP */
        if (j<i) /* COPY TO NEW COMPACTED LOCATION */
          memcpy(im+j,im+i,sizeof(IntervalMap));
        j++;
      }

    memcpy(im+j,imsub,nsub*sizeof(IntervalMap)); /* COPY THE SUBLISTS */
    for (i=0;i<nlists;i++) /* ADJUST start ADDRESSES FOR SHIFT*/
      subheader[i].start += j;
    FREE(imsub);
    *p_n = j; /* COPY THE COMPRESSED LIST SIZES BACK TO CALLER*/
  }
  else {  /* NO SUBLISTS: HANDLE THIS CASE CAREFULLY */
    *p_n = n;
    CALLOC(subheader,1,SublistHeader); /* RETURN A DUMMY ARRAY, SINCE NULL RETURN IS ERROR CODE */
  }
  *p_nlists=nlists; /* RETURN COUNT OF NUMBER OF SUBLISTS */
  return subheader;
 handle_malloc_failure:
  FREE(imsub);  /* FREE ANY MALLOCS WE PERFORMED*/
  FREE(subheader);
  return NULL;
}

int *alloc_array(int n){
  /* var = (int*) malloc(n * sizeof(int)); */
  int *arr = NULL;
  CALLOC(arr,n,int);
  handle_malloc_failure:
    return NULL;
};


IntervalMap *interval_map_alloc(int n)
{
  IntervalMap *im=NULL;
  CALLOC(im,n,IntervalMap);
  return im;
 handle_malloc_failure:
  return NULL;
}



inline int find_overlap_start(int start,int end,IntervalMap im[],int n)
{
  int l=0,mid,r;

  r=n-1;
  while (l<r) {
    mid=(l+r)/2;
    if (END_POSITIVE(im[mid])<=start)
      l=mid+1;
    else
      r=mid;
  }
  if (l<n && HAS_OVERLAP_POSITIVE(im[l],start,end))
    return l; /* l IS START OF OVERLAP */
  else
    return -1; /* NO OVERLAP FOUND */
}




int find_index_start(int start,int end,IntervalIndex im[],int n)
{
  int l=0,mid,r;

  r=n-1;
  while (l<r) {
    mid=(l+r)/2;
    if (END_POSITIVE(im[mid])<=start)
      l=mid+1;
    else
      r=mid;
  }
  return l; /* l IS START OF POSSIBLE OVERLAP */
}



inline int find_suboverlap_start(int start,int end,int isub,IntervalMap im[],
                                 SublistHeader subheader[])
{
  int i;

  if (isub>=0) {
    i=find_overlap_start(start,end,im+subheader[isub].start,subheader[isub].len);
    if (i>=0)
      return i+subheader[isub].start;
  }
  return -1;
}


IntervalIterator *interval_iterator_alloc(void)
{
  IntervalIterator *it=NULL;
  CALLOC(it,1,IntervalIterator);
  return it;
 handle_malloc_failure:
  return NULL;
}

int free_interval_iterator(IntervalIterator *it)
{
  IntervalIterator *it2,*it_next;
  if (!it)
    return 0;
  FREE_ITERATOR_STACK(it,it2,it_next);
  return 0;
}


int find_intervals(IntervalIterator *it0, int start, int end,
                   IntervalMap im[],int n,
                   SublistHeader subheader[], int nlists,
                   IntervalMap buf[], int nbuf,
                   int *p_nreturn, IntervalIterator **it_return)
{
  IntervalIterator *it=NULL,*it2=NULL;
  int ibuf=0,j,k,ori_sign=1;
  if (!it0) { /* ALLOCATE AN ITERATOR IF NOT SUPPLIED*/
    CALLOC(it,1,IntervalIterator);
  }
  else
    it=it0;

#if defined(ALL_POSITIVE_ORIENTATION) || defined(MERGE_INTERVAL_ORIENTATIONS)
  if (start<0) { /* NEED TO CONVERT TO POSITIVE ORIENTATION */
    j=start;
    start= -end;
    end= -j;
    ori_sign = -1;
  }
#endif
  if (it->n == 0) { /* DEFAULT: SEARCH THE TOP NESTED LIST */
    it->n=n;
    it->i=find_overlap_start(start,end,im,n);
  }

  do {
    while (it->i>=0 && it->i<it->n && HAS_OVERLAP_POSITIVE(im[it->i],start,end)) {
      memcpy(buf+ibuf,im + it->i,sizeof(IntervalMap)); /*SAVE THIS HIT TO BUFFER */
      ibuf++;
      k=im[it->i].sublist; /* GET SUBLIST OF i IF ANY */
      it->i++; /* ADVANCE TO NEXT INTERVAL */
      if (k>=0 && (j=find_suboverlap_start(start,end,k,im,subheader))>=0) {
        PUSH_ITERATOR_STACK(it,it2,IntervalIterator); /* RECURSE TO SUBLIST */
        it2->i = j; /* START OF OVERLAPPING HITS IN THIS SUBLIST */
        it2->n = subheader[k].start+subheader[k].len; /* END OF SUBLIST */
        it=it2; /* PUSH THE ITERATOR STACK */
      }
      if (ibuf>=nbuf){ /* FILLED THE BUFFER, RETURN THE RESULTS SO FAR */
        goto finally_return_result;
      }
    }
  } while (POP_ITERATOR_STACK(it));  /* IF STACK EXHAUSTED,  EXIT */
  if (!it0) /* FREE THE ITERATOR WE CREATED.  NO NEED TO RETURN IT TO USER */
    free_interval_iterator(it);
  it=NULL;  /* ITERATOR IS EXHAUSTED */

 finally_return_result:
/* #if defined(ALL_POSITIVE_ORIENTATION) || defined(MERGE_INTERVAL_ORIENTATIONS) */
/*   reorient_intervals(ibuf,buf,ori_sign); /\* REORIENT INTERVALS TO MATCH QUERY ORI *\/ */
/* #endif */

  *p_nreturn=ibuf; /* #INTERVALS FOUND IN THIS PASS */
  *it_return=it; /* HAND BACK ITERATOR FOR CONTINUING THE SEARCH, IF ANY */
  return 0; /* SIGNAL THAT NO ERROR OCCURRED */
 handle_malloc_failure:
  return -1;
}



void reorient_intervals(int n,IntervalMap im[],int ori_sign)
{
  int i,tmp;
  for (i=0;i<n;i++) {
    if ((im[i].start>=0 ? 1:-1)!=ori_sign) { /* ORIENTATION MISMATCH */
      tmp=im[i].start; /* SO REVERSE THIS INTERVAL MAPPING */
      im[i].start= -im[i].end;
      im[i].end =  -tmp;
      /* tmp=im[i].target_start; */
      /* im[i].target_start= -im[i].target_end; */
      /* im[i].target_end =  -tmp; */
    }
  }
}

IntervalIterator *reset_interval_iterator(IntervalIterator *it)
{
  ITERATOR_STACK_TOP(it);
  it->n=0;
  return it;
}

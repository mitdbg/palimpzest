

#include <stdint.h>
#include "fintervaldb.h"

int C_int_max=INT_MAX; /* KLUDGE TO LET PYREX CODE ACCESS VALUE OF INT_MAX MACRO */


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



/* int target_qsort_cmp(const void *void_a,const void *void_b) */
/* { /\* SORT IN target_id ORDER, SECONDARILY BY target_start *\/ */
/*   IntervalMap *a=(IntervalMap *)void_a,*b=(IntervalMap *)void_b; */
/*   if (a->target_id<b->target_id) */
/*     return -1; */
/*   else if (a->target_id>b->target_id) */
/*     return 1; */
/*   else if (a->target_start < b->target_start) */
/*     return -1; */
/*   else if (a->target_start > b->target_start) */
/*     return 1; */
/*   else */
/*     return 0; */
/* } */


SublistHeader *build_nested_list_inplace(IntervalMap im[],int n,
                                         int *p_n,int *p_nlists)
{
  int i=0,parent,nlists=1,isublist=0,total=0,temp=0;
  SublistHeader *subheader=NULL;

#ifdef ALL_POSITIVE_ORIENTATION
  reorient_intervals(n,im,1); /* FORCE ALL INTERVALS INTO POSITIVE ORI */
#endif
#ifdef MERGE_INTERVAL_ORIENTATIONS
  qsort(im,n,sizeof(IntervalMap),im_qsort_cmp); /* SORT BY start, CONTAINMENT */
#else
  qsort(im,n,sizeof(IntervalMap),imstart_qsort_cmp); /* SORT BY start, CONTAINMENT */
#endif
  nlists=1;
  for(i=1;i<n;++i){
    if(!(END_POSITIVE(im[i])>END_POSITIVE(im[i-1]) /* i NOT CONTAINED */
         || (END_POSITIVE(im[i])==END_POSITIVE(im[i-1]) /* SAME INTERVAL! */
             && START_POSITIVE(im[i])==START_POSITIVE(im[i-1])))){
      nlists++;
      /*       printf("%d (%d,%d) -> (%d,%d) %d\n", nlists, im[i-1].start, */
      /* 	     im[i-1].end, im[i].start,im[i].end,i); */
    }
  }

  /*   printf("%d lists?!\n", nlists); */
  *p_nlists=nlists-1;

  if(nlists==1){
    *p_n=n;
    CALLOC(subheader,1,SublistHeader); /* RETURN A DUMMY ARRAY, SINCE NULL RETURN IS ERROR CODE */
    return subheader;
  }

  CALLOC(subheader,nlists+1,SublistHeader); /* SUBLIST HEADER INDEX */

  im[0].sublist=0;
  subheader[0].start= -1;
  subheader[0].len=1;
  parent=0;
  nlists=1;
  isublist=1;
  for(i=1;i<n;){
    if(isublist && (END_POSITIVE(im[i])>END_POSITIVE(im[parent]) /* i NOT CONTAINED */
                    || (END_POSITIVE(im[i])==END_POSITIVE(im[parent]) /* SAME INTERVAL! */
                        && START_POSITIVE(im[i])==START_POSITIVE(im[parent])))){
      subheader[isublist].start=subheader[im[parent].sublist].len-1; /* RECORD PARENT RELATIVE POSITION */
      isublist=im[parent].sublist;
      parent=subheader[im[parent].sublist].start;
    }
    else{
      if(subheader[isublist].len==0){
        nlists++;
      }
      subheader[isublist].len++;
      im[i].sublist=isublist;
      parent=i;
      isublist=nlists;
      subheader[isublist].start=parent;
      i++;
    }
  }

  while(isublist>0){ /* pop remaining stack */
    subheader[isublist].start=subheader[im[parent].sublist].len-1; /* RECORD PARENT RELATIVE POSITION */
    isublist=im[parent].sublist;
    parent=subheader[im[parent].sublist].start;
  }

  *p_n=subheader[0].len;

  total=0;
  for(i=0;i<nlists+1;++i){
    temp=subheader[i].len;
    subheader[i].len=total;
    total+=temp;
  };

  /* SUBHEADER.LEN IS NOW START OF THE SUBLIST */

  for(i=1;i<n;i+=1){
    if(im[i].sublist>im[i-1].sublist){
      subheader[im[i].sublist].start+=subheader[im[i-1].sublist].len;
    }
  }

  /* SUBHEADER.START IS NOW ABS POSITION OF PARENT */

  qsort(im,n,sizeof(IntervalMap),sublist_qsort_cmp);
  /* AT THIS POINT SUBLISTS ARE GROUPED TOGETHER, READY TO PACK */

  isublist=0;
  subheader[0].start=0;
  subheader[0].len=0;
  for(i=0;i<n;++i){
    if(im[i].sublist>isublist){
      /*       printf("Entering sublist %d (%d,%d)\n", im[i].sublist, im[i].start,im[i].end); */
      isublist=im[i].sublist;
      parent=subheader[isublist].start;
      /*       printf("Parent (%d,%d) is at %d, list start is at %d\n",  */
      /* 	     im[parent].start, im[parent].end, subheader[isublist].start,i); */
      im[parent].sublist=isublist-1;
      subheader[isublist].len=0;
      subheader[isublist].start=i;
    }
    subheader[isublist].len++;
    im[i].sublist= -1;
  }

  nlists--;
  memmove(subheader,subheader+1,nlists*sizeof(SublistHeader));

  return subheader;
 handle_malloc_failure:
  /* FREE ANY MALLOCS WE PERFORMED*/
  FREE(subheader);
  return NULL;
}



SublistHeader *build_nested_list(IntervalMap im[],int n,
                                 int *p_n,int *p_nlists)
{
  int i=0,j,k,parent,nsub=0,nlists=0;
  IntervalMap *imsub=NULL;
  SublistHeader *subheader=NULL;

#ifdef ALL_POSITIVE_ORIENTATION
  reorient_intervals(n,im,1); /* FORCE ALL INTERVALS INTO POSITIVE ORI */
#endif
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



inline int64_t find_overlap_start(double start,double end,IntervalMap im[],int n)
{
  int64_t l=0, mid, r;

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



inline double find_suboverlap_start(double start,double end, int64_t isub, IntervalMap im[],
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


IntervalIterator *reset_interval_iterator(IntervalIterator *it)
{
  ITERATOR_STACK_TOP(it);
  it->n=0;
  return it;
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


double find_intervals(IntervalIterator *it0, double start, double end,
                   IntervalMap im[],int n,
                   SublistHeader subheader[], int nlists,
                   IntervalMap buf[], int nbuf,
                   int *p_nreturn, IntervalIterator **it_return)
{
  IntervalIterator *it=NULL,*it2=NULL;
  int64_t ibuf=0,j,k,ori_sign=1;
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
#if defined(ALL_POSITIVE_ORIENTATION) || defined(MERGE_INTERVAL_ORIENTATIONS)
  reorient_intervals(ibuf,buf,ori_sign); /* REORIENT INTERVALS TO MATCH QUERY ORI */
#endif

  *p_nreturn=ibuf; /* #INTERVALS FOUND IN THIS PASS */
  *it_return=it; /* HAND BACK ITERATOR FOR CONTINUING THE SEARCH, IF ANY */
  return 0; /* SIGNAL THAT NO ERROR OCCURRED */
 handle_malloc_failure:
  return -1;
}





/****************************************************************
 *
 *   FILE-BASED SEARCH FUNCTIONS
 */


/* READ A BLOCK FROM THE DATABASE FILE */
int read_imdiv(FILE *ifile,IntervalMap imdiv[],int div,int i_div,int ntop)
{
  int block;
  PYGR_OFF_T ipos;
  ipos=div*i_div; /* CALCULATE POSITION IN RECORDS */
  if (ipos+div<=ntop) /* GET A WHOLE BLOCK */
    block=div;
  else /* JUST READ PARTIAL BLOCK AT END */
    block=ntop%div;
  ipos *= sizeof(IntervalMap); /* CALCULATE FILE POSITION IN BYTES */
  PYGR_FSEEK(ifile,ipos,SEEK_SET);
  fread(imdiv,sizeof(IntervalMap),block,ifile);
  return block;
}


/* READ A SUBLIST FROM DATABASE FILE */
IntervalMap *read_sublist(FILE *ifile,SublistHeader *subheader,
                          IntervalMap *im)
{
  PYGR_OFF_T ipos;
  if (im==NULL) {
    CALLOC(im,subheader->len,IntervalMap);
  }
  ipos=subheader->start; /* CALCULATE POSITION IN RECORDS */
  ipos*=sizeof(IntervalMap);  /* CALCULATE FILE POSITION IN BYTES */
  PYGR_FSEEK(ifile,ipos,SEEK_SET);
  fread(im,sizeof(IntervalMap),subheader->len,ifile);
  return im;
 handle_malloc_failure:
  return NULL;
}


/* READ A BLOCK OF THE SUBLIST HEADER FILE */
int read_subheader_block(SublistHeader subheader[],int isub,int nblock,
                         int nsubheader,FILE *ifile)
{
  PYGR_OFF_T ipos;
  long start;
  start=isub-(isub%nblock); /* GET BLOCK START */
  if (start+nblock>nsubheader)
    nblock=nsubheader-start; /* TRUNCATE TO FIT MAX FILE LENGTH */
  ipos=start; /* CONVERT TO off_t TYPE */
  ipos *= sizeof(SublistHeader); /* CALCULATE ACTUAL BYTE OFFSET */
  PYGR_FSEEK(ifile,ipos,SEEK_SET);
  fread(subheader,sizeof(SublistHeader),nblock,ifile);
  return start;
}




int find_file_start(IntervalIterator *it,int start,int end,int isub,
                    IntervalIndex ii[],int nii,
                    SublistHeader *subheader,int nlists,
                    SubheaderFile *subheader_file,
                    int ntop,int div,FILE *ifile)
{
  int i_div= -1,offset=0,offset_div=0;
  if (isub<0)  /* TOP-LEVEL SEARCH: USE THE INDEX */
    i_div=find_index_start(start,end,ii,nii);
  else { /* GET PTR TO subheader[isub] */
#ifdef ON_DEMAND_SUBLIST_HEADER
    if (isub<subheader_file->start /* isub OUTSIDE OUR CURRENT BLOCK */
        || isub>=subheader_file->start+subheader_file->nblock)
      subheader_file->start=  /* LOAD NEW BLOCK FROM DISK */
        read_subheader_block(subheader_file->subheader,isub,
                             subheader_file->nblock,nlists,
                             subheader_file->ifile);
    subheader=subheader_file->subheader + (isub-subheader_file->start);
#else
    subheader += isub; /* POINT TO OUR SUBHEADER */
#endif
    if (subheader->len>div) { /* BIG SUBLIST, SO USE THE INDEX */
      offset=subheader->start;
      offset_div=offset/div;/* offset GUARANTEED TO BE MULTIPLE OF div */
      ntop=subheader->len;
      nii=ntop/div; /* CALCULATE SUBLIST INDEX SIZE */
      if (ntop%div) /* ONE EXTRA ENTRY FOR PARTIAL BLOCK */
        nii++;
      i_div=find_index_start(start,end,ii+offset_div,nii);
    }
  }

  if (!it->im) { /* NO ALLOCATION? ALLOCATE OUR BLOCK SIZE div */
    CALLOC(it->im,div,IntervalMap); /* ALWAYS ALLOCATE div BUFFERSIZE */
  }
  if (i_div>=0) { /* READ A SPECIFIC BLOCK OF SIZE div */
    it->n=read_imdiv(ifile,it->im,div,i_div+offset_div,ntop+offset);
    it->ntop=ntop+offset; /* END OF THIS LIST IN THE BINARY FILE */
    it->nii=nii+offset_div; /* SAVE INFORMATION FOR READING SUBSEQUENT BLOCKS */
    it->i_div=i_div+offset_div; /* INDEX OF THIS BLOCK IN THE BINARY FILE */
  }
  else { /* A SMALL SUBLIST: READ THE WHOLE LIST INTO MEMORY */
    read_sublist(ifile,subheader,it->im); /* GUARANTEED TO BE <=div ITEMS */
    it->n=subheader->len;
    it->nii=1;
    it->i_div=0; /* INDICATE THAT THERE ARE NO ADDITIONAL BLOCKS TO READ*/
  }

  it->i=find_overlap_start(start,end,it->im,it->n);
  return it->i;
 handle_malloc_failure:
  return -2; /* SIGNAL THAT MEMORY ERROR OCCURRED */
}


int find_file_intervals(IntervalIterator *it0,int start,int end,
                        IntervalIndex ii[],int nii,
                        SublistHeader subheader[],int nlists,
                        SubheaderFile *subheader_file,
                        int ntop,int div,FILE *ifile,
                        IntervalMap buf[],int nbuf,
                        int *p_nreturn,IntervalIterator **it_return)
{
  IntervalIterator *it=NULL,*it2=NULL;
  int k,ibuf=0,ori_sign=1,ov=0;
  if (!it0) { /* ALLOCATE AN ITERATOR IF NOT SUPPLIED*/
    CALLOC(it,1,IntervalIterator);
  }
  else
    it=it0;

#if defined(ALL_POSITIVE_ORIENTATION) || defined(MERGE_INTERVAL_ORIENTATIONS)
  if (start<0) { /* NEED TO CONVERT TO POSITIVE ORIENTATION */
    k=start;
    start= -end;
    end= -k;
    ori_sign = -1;
  }
#endif

  if (it->n == 0)  /* DEFAULT: SEARCH THE TOP NESTED LIST */
    if (find_file_start(it,start,end,-1,ii,nii,subheader,nlists,
                        subheader_file,ntop,div,ifile) == FIND_FILE_MALLOC_ERR)
      goto handle_malloc_failure;

  do { /* ITERATOR STACK LOOP */
    while (it->i_div < it->nii) { /* BLOCK ITERATION LOOP */
      while (it->i>=0 && it->i<it->n /* INDIVIDUAL INTERVAL ITERATION LOOP */
             && HAS_OVERLAP_POSITIVE(it->im[it->i],start,end)) { /*OVERLAPS!*/
        memcpy(buf+ibuf,it->im + it->i,sizeof(IntervalMap)); /*SAVE THIS HIT */
        ibuf++;
        k=it->im[it->i].sublist; /* GET SUBLIST OF i IF ANY */
        it->i++; /* ADVANCE TO NEXT INTERVAL */
        PUSH_ITERATOR_STACK(it,it2,IntervalIterator); /* RECURSE TO SUBLIST */
        if (k>=0 && (ov=find_file_start(it2,start,end,k,ii,nii,subheader,nlists,
                                        subheader_file,ntop,div,ifile))>=0)
          it=it2; /* PUSH THE ITERATOR STACK */
        if (FIND_FILE_MALLOC_ERR == ov)
          goto handle_malloc_failure;

        if (ibuf>=nbuf)  /* FILLED THE BUFFER, RETURN THE RESULTS SO FAR */
          goto finally_return_result;
      }
      it->i_div++; /* TRY GOING TO NEXT BLOCK */
      if (it->i == it->n  /* USED WHOLE BLOCK, SO THERE MIGHT BE MORE */
          && it->i_div < it->nii) { /* CONTINUE TO NEXT BLOCK */
        it->n=read_imdiv(ifile,it->im,div,it->i_div,it->ntop); /*READ NEXT BLOCK*/
        it->i=0; /* PROCESS IT FROM ITS START */
      }
    }
  } while (POP_ITERATOR_STACK(it));  /* IF STACK EXHAUSTED,  EXIT */
  if (!it0) /* FREE THE ITERATOR WE CREATED.  NO NEED TO RETURN IT TO USER */
    free_interval_iterator(it);
  it=NULL;  /* ITERATOR IS EXHAUSTED */

 finally_return_result:
#if defined(ALL_POSITIVE_ORIENTATION) || defined(MERGE_INTERVAL_ORIENTATIONS)
  reorient_intervals(ibuf,buf,ori_sign); /* REORIENT INTERVALS TO MATCH QUERY ORI */
#endif
  *p_nreturn=ibuf; /* #INTERVALS FOUND IN THIS PASS */
  *it_return=it; /* HAND BACK ITERATOR FOR CONTINUING THE SEARCH, IF ANY */
  return 0; /* SIGNAL THAT NO ERROR OCCURRED */
 handle_malloc_failure:
  return -1;
}






/* FUNCTIONS FOR READING AND WRITING OF THE BINARY DATABASE FILES */

int write_padded_binary(IntervalMap im[],int n,int div,FILE *ifile)
{
  int i,npad;
  fwrite(im,sizeof(IntervalMap),n,ifile); /* SAVE THE ACTUAL DATA */
  npad=n%div;
  if (npad) {
    npad=div-npad; /* #ITEMS NEEDED TO PAD TO AN EXACT MULTIPLE OF div */
    for (i=0;i<npad;i++) /* GUARANTEED im HAS AT LEAST ONE RECORD */
      fwrite(im,sizeof(IntervalMap),1,ifile); /*THIS IS JUST PADDING */
  }
  return n+npad; /* #RECORDS SAVED */
}


int repack_subheaders(IntervalMap im[],int n,int div,
                      SublistHeader *subheader,int nlists)
{
  int i,j,*sub_map=NULL;
  SublistHeader *sub_pack=NULL;

  CALLOC(sub_map,nlists,int);
  CALLOC(sub_pack,nlists,SublistHeader);
  for (i=j=0;i<nlists;i++) { /* PLACE SUBLISTS W/ len>div AT FRONT */
    if (subheader[i].len>div) {
      memcpy(sub_pack+j,subheader+i,sizeof(SublistHeader));
      sub_map[i]=j;
      j++;
    }
  }
  for (i=0;i<nlists;i++) { /* PLACE SUBLISTS W/ len<=div AFTERWARDS */
    if (subheader[i].len<=div) {
      memcpy(sub_pack+j,subheader+i,sizeof(SublistHeader));
      sub_map[i]=j;
      j++;
    }
  }
  for (i=0;i<n;i++) /* ADJUST im[].sublist TO THE NEW LOCATIONS */
    if (im[i].sublist>=0)
      im[i].sublist=sub_map[im[i].sublist];
  memcpy(subheader,sub_pack,nlists*sizeof(SublistHeader)); /* SAVE REORDERED LIST*/

  FREE(sub_map);
  FREE(sub_pack);
  return 0;
 handle_malloc_failure:
  return -1;
}


int write_binary_index(IntervalMap im[],int n,int div,FILE *ifile)
{
  int i,j,nsave=0;
  for (i=0;i<n;i+=div) {
#ifdef MERGE_INTERVAL_ORIENTATIONS
    if (im[i].start>=0) /* FORWARD ORI */
#endif
      fwrite(&(im[i].start),sizeof(int),1,ifile);  /*SAVE start */
#ifdef MERGE_INTERVAL_ORIENTATIONS
    else { /* REVERSE ORI */
      j= - im[i].end;
      fwrite(&j,sizeof(int),1,ifile);  /*SAVE start */
    }
#endif
    j=i+div-1;
    if (j>=n)
      j=n-1;
#ifdef MERGE_INTERVAL_ORIENTATIONS
    if (im[j].start>=0)  /* FORWARD ORI */
#endif
      fwrite(&(im[j].end),sizeof(int),1,ifile);  /*SAVE end */
#ifdef MERGE_INTERVAL_ORIENTATIONS
    else { /* REVERSE ORI */
      j= - im[j].start;
      fwrite(&j,sizeof(int),1,ifile);  /*SAVE end */
    }
#endif
    nsave++;
  }
  return nsave;
}



char *write_binary_files(IntervalMap im[],int n,int ntop,int div,
                         SublistHeader *subheader,int nlists,char filestem[])
{
  int i,npad=0,nii;
  char path[2048];
  FILE *ifile=NULL,*ifile_subheader=NULL;
  SublistHeader sh_tmp;
  static char err_msg[1024];

  if (nlists>0  /* REPACK SMALL SUBLISTS TO END */
      && repack_subheaders(im,n,div,subheader,nlists)
      == FIND_FILE_MALLOC_ERR) {
    sprintf(err_msg,"unable to malloc %d subheaders",nlists);
    return err_msg;
  }
  sprintf(path,"%s.subhead",filestem); /* SAVE THE SUBHEADER LIST */
  ifile_subheader=fopen(path,"wb"); /* binary file */
  if (!ifile_subheader) {
    sprintf(err_msg,"unable to open file %s for writing",path);
    return err_msg;
  }
  sprintf(path,"%s.idb",filestem); /* SAVE THE DATABASE */
  ifile=fopen(path,"wb"); /* binary file */
  if (!ifile) {
    sprintf(err_msg,"unable to open file %s for writing",path);
    return err_msg;
  }
  npad=write_padded_binary(im,ntop,div,ifile); /* WRITE THE TOP LEVEL LIST */
  for (i=0;i<nlists;i++) {
    sh_tmp.start=npad; /* FILE LOCATION WHERE THIS SUBLIST STORED */
    sh_tmp.len=subheader[i].len; /* SAVE THE TRUE SUBLIST LENGTH, UNPADDED */
    fwrite(&sh_tmp,sizeof(SublistHeader),1,ifile_subheader);
    if (subheader[i].len>div) /* BIG LIST: PAD TO EXACT MULTIPLE OF div */
      npad+=write_padded_binary(im+subheader[i].start,subheader[i].len,div,ifile);
    else { /* SMALL LIST: SAVE W/O PADDING */
      fwrite(im+subheader[i].start,sizeof(IntervalMap),subheader[i].len,ifile);
      npad+=subheader[i].len;
    }
  }
  fclose(ifile);
  fclose(ifile_subheader);

  sprintf(path,"%s.index",filestem); /* SAVE THE COMPACTED INDEX */
  ifile=fopen(path,"wb"); /* binary file */
  if (!ifile) {
    sprintf(err_msg,"unable to open file %s for writing",path);
    return err_msg;
  }
  nii=write_binary_index(im,ntop,div,ifile);
  for (i=0;i<nlists;i++) /* ALSO STORE INDEX DATA FOR BIG SUBLISTS */
    if (subheader[i].len>div)
      nii+=write_binary_index(im+subheader[i].start,subheader[i].len,div,ifile);
  fclose(ifile);

  sprintf(path,"%s.size",filestem); /* SAVE BASIC SIZE INFO*/
  ifile=fopen(path,"w"); /* text file */
  if (!ifile) {
    sprintf(err_msg,"unable to open file %s for writing",path);
    return err_msg;
  }
  fprintf(ifile,"%d %d %d %d %d\n",n,ntop,div,nlists,nii);
  fclose(ifile);

  return NULL; /* RETURN CODE SIGNALS SUCCESS!! */
}



IntervalDBFile *read_binary_files(char filestem[],char err_msg[],
                                  int subheader_nblock)
{
  int n,ntop,div,nlists,nii;
  char path[2048];
  IntervalIndex *ii=NULL;
  SublistHeader *subheader=NULL;
  IntervalDBFile *idb_file=NULL;
  FILE *ifile=NULL;

  sprintf(path,"%s.size",filestem); /* READ BASIC SIZE INFO*/
  ifile=fopen(path,"r"); /* text file */
  if (!ifile) {
    if (err_msg)
      sprintf(err_msg,"unable to open file %s",path);
    return NULL;
  }
  fscanf(ifile,"%d %d %d %d %d",&n,&ntop,&div,&nlists,&nii);
  fclose(ifile);

  CALLOC(ii,nii+1,IntervalIndex);
  if (nii>0) {
    sprintf(path,"%s.index",filestem); /* READ THE COMPACTED INDEX */
    ifile=fopen(path,"rb"); /* binary file */
    if (!ifile) {
      if (err_msg)
        sprintf(err_msg,"unable to open file %s",path);
      return NULL;
    }
    fread(ii,sizeof(IntervalIndex),nii,ifile);
    fclose(ifile);
  }

  CALLOC(idb_file,1,IntervalDBFile);
  if(nlists>0){
    sprintf(path,"%s.subhead",filestem); /* SAVE THE SUBHEADER LIST */
    ifile=fopen(path,"rb"); /* binary file */
    if (!ifile) {
      if (err_msg)
        sprintf(err_msg,"unable to open file %s",path);
      return NULL;
    }
#ifdef ON_DEMAND_SUBLIST_HEADER
    CALLOC(subheader,subheader_nblock,SublistHeader);
    idb_file->subheader_file.subheader=subheader;
    idb_file->subheader_file.nblock=subheader_nblock;
    idb_file->subheader_file.start = -subheader_nblock; /* NO BLOCK LOADED */
    idb_file->subheader_file.ifile=ifile;
#else
    CALLOC(subheader,nlists,SublistHeader); /* LOAD THE ENTIRE SUBHEADER */
    fread(subheader,sizeof(SublistHeader),nlists,ifile);  /*SAVE LIST */
    fclose(ifile);
#endif
  }

  idb_file->n=n;
  idb_file->ntop=ntop;
  idb_file->nlists=nlists;
  idb_file->div=div;
  idb_file->nii=ntop/div;
  if (ntop%div) /* INDEX IS PADDED TO EXACT MULTIPLE OF div */
    idb_file->nii++; /* ONE EXTRA ENTRY FOR PARTIAL BLOCK */
  idb_file->ii=ii;
  idb_file->subheader=subheader;
  sprintf(path,"%s.idb",filestem); /* OPEN THE DATABASE */
  idb_file->ifile_idb=fopen(path,"rb"); /* binary file */
  if (!idb_file->ifile_idb) {
    if (err_msg)
      sprintf(err_msg,"unable to open file %s",path);
    free(idb_file);
    return NULL;
  }
  return idb_file;
 handle_malloc_failure:
  FREE(ii); /* DUMP OUR MEMORY */
  FREE(subheader);
  FREE(idb_file);
  return NULL;
}



int free_interval_dbfile(IntervalDBFile *db_file)
{
  if (db_file->ifile_idb)
    fclose(db_file->ifile_idb);
#ifdef ON_DEMAND_SUBLIST_HEADER
  if (db_file->subheader_file.ifile)
    fclose(db_file->subheader_file.ifile);
#endif
  FREE(db_file->ii);
  FREE(db_file->subheader);
  free(db_file);
  return 0;
}




/* int save_text_file(char filestem[],char basestem[], */
/* 		   char err_msg[],FILE *ofile) */
/* { */
/*   int i,n,ntop,div,nlists,nii,npad; */
/*   char path[2048]; */
/*   IntervalMap im; */
/*   IntervalIndex ii; */
/*   SublistHeader subheader; */
/*   FILE *ifile=NULL; */

/*   sprintf(path,"%s.size",filestem); /\* READ BASIC SIZE INFO*\/ */
/*   ifile=fopen(path,"r"); /\* text file *\/ */
/*   if (!ifile) */
/*     goto unable_to_open_file; */
/*   if (5!=fscanf(ifile,"%d %d %d %d %d",&n,&ntop,&div,&nlists,&nii)) */
/*     goto fread_error_occurred; */
/*   fclose(ifile); */
/*   npad=ntop%div; */
/*   if (npad>0) /\* PAD TO AN EXACT MULTIPLE OF div *\/ */
/*     npad=ntop+(div-npad); */
/*   else /\* AN EXACT MULTIPLE OF div, SO NO PADDING *\/ */
/*     npad=ntop; */

/*   if (fprintf(ofile,"SIZE\t%s\t%d %d %d %d %d\n", */
/* 	      basestem,n,ntop,div,nlists,nii)<0) */
/*     goto write_error_occurred; */

/*   if (nii>0) { */
/*     sprintf(path,"%s.index",filestem); /\* READ THE COMPACTED INDEX *\/ */
/*     ifile=fopen(path,"rb"); /\* binary file *\/ */
/*     if (!ifile) */
/*       goto unable_to_open_file; */
/*     for (i=0;i<nii;i++) { */
/*       if (1!=fread(&ii,sizeof(IntervalIndex),1,ifile)) */
/* 	goto fread_error_occurred; */
/*       if (fprintf(ofile,"I %d %d\n",ii.start,ii.end)<0) */
/* 	goto write_error_occurred; */
/*     } */
/*     fclose(ifile); */
/*   } */

/*   if(nlists>0){ */
/*     sprintf(path,"%s.subhead",filestem); /\* READ THE SUBHEADER LIST *\/ */
/*     ifile=fopen(path,"rb"); /\* binary file *\/ */
/*     if (!ifile) */
/*       goto unable_to_open_file; */
/*     for (i=0;i<nlists;i++) { */
/*       if (1!=fread(&subheader,sizeof(SublistHeader),1,ifile)) */
/* 	goto fread_error_occurred; */
/*       if (fprintf(ofile,"S %d %d\n",subheader.start,subheader.len)<0) */
/* 	goto write_error_occurred; */
/*       npad=subheader.start+subheader.len; */
/*     } */
/*     fclose(ifile); */
/*   } */

/*   if (npad>0) { */
/*     sprintf(path,"%s.idb",filestem); /\* READ THE DATABASE *\/ */
/*     ifile=fopen(path,"rb"); /\* binary file *\/ */
/*     if (!ifile) */
/*       goto unable_to_open_file; */
/*     for (i=0;i<npad;i++) { */
/*       if (1!=fread(&im,sizeof(IntervalMap),1,ifile)) */
/* 	goto fread_error_occurred; */
/*       if (fprintf(ofile,"M %d %d %d %d %d %d\n",im.start,im.end, */
/* 		  im.target_id,im.target_start, */
/* 		  im.target_end,im.sublist)<0) */
/* 	goto write_error_occurred; */
/*     } */
/*     fclose(ifile); */
/*   } */
/*   return 0; /\* INDICATES NO ERROR OCCURRED *\/ */
/*  unable_to_open_file: */
/*   if (err_msg) */
/*     sprintf(err_msg,"unable to open file %s",path); */
/*   return -1; */
/*  fread_error_occurred: */
/*   if (err_msg) */
/*     sprintf(err_msg,"error or EOF reading file %s",path); */
/*   return -1; */
/*  write_error_occurred: */
/*   if (err_msg) */
/*     sprintf(err_msg,"error writing output file! out of disk space?"); */
/*   return -1; */
/* } */



/* int text_file_to_binaries(FILE *infile,char buildpath[],char err_msg[]) */
/* { */
/*   int i,n,ntop,div,nlists,nii,npad; */
/*   char path[2048],line[32768],filestem[2048]; */
/*   IntervalMap im; */
/*   IntervalIndex ii; */
/*   SublistHeader subheader; */
/*   FILE *ifile=NULL; */

/*   if (NULL==fgets(line,32767,infile)) */
/*     goto fread_error_occurred; */
/*   if (6!=sscanf(line,"SIZE\t%s\t%d %d %d %d %d", */
/* 		filestem,&n,&ntop,&div,&nlists,&nii)) */
/*     goto fread_error_occurred; */
/*   sprintf(path,"%s%s.size",buildpath,filestem); /\* SAVE BASIC SIZE INFO*\/ */
/*   ifile=fopen(path,"w"); /\* text file *\/ */
/*   if (!ifile) */
/*     goto unable_to_open_file; */
/*   if (fprintf(ifile,"%d %d %d %d %d\n",n,ntop,div,nlists,nii)<0) */
/*     goto write_error_occurred; */
/*   fclose(ifile); */
/*   npad=ntop%div; */
/*   if (npad>0) /\* PAD TO AN EXACT MULTIPLE OF div *\/ */
/*     npad=ntop+(div-npad); */
/*   else /\* AN EXACT MULTIPLE OF div, SO NO PADDING *\/ */
/*     npad=ntop; */

/*   if (nii>0) { */
/*     sprintf(path,"%s%s.index",buildpath,filestem); /\* SAVE INDEX INFO*\/ */
/*     ifile=fopen(path,"wb"); /\* binary file *\/ */
/*     if (!ifile) */
/*       goto unable_to_open_file; */
/*     for (i=0;i<nii;i++) { */
/*       if (NULL==fgets(line,32767,infile)) */
/* 	goto fread_error_occurred; */
/*       if (2!=sscanf(line,"I %d %d",&(ii.start),&(ii.end))) */
/* 	goto fread_error_occurred; */
/*       if (1!=fwrite(&ii,sizeof(IntervalIndex),1,ifile)) */
/* 	goto write_error_occurred; */
/*     } */
/*     fclose(ifile); */
/*   } */

/*   if(nlists>0){ */
/*     sprintf(path,"%s%s.subhead",buildpath,filestem); /\* SAVE THE SUBHEADER LIST *\/ */
/*     ifile=fopen(path,"wb"); /\* binary file *\/ */
/*     if (!ifile) */
/*       goto unable_to_open_file; */
/*     for (i=0;i<nlists;i++) { */
/*       if (NULL==fgets(line,32767,infile)) */
/* 	goto fread_error_occurred; */
/*       if (2!=sscanf(line,"S %d %d",&(subheader.start),&(subheader.len))) */
/* 	goto fread_error_occurred; */
/*       if (1!=fwrite(&subheader,sizeof(SublistHeader),1,ifile)) */
/* 	goto write_error_occurred; */
/*       npad=subheader.start+subheader.len; */
/*     } */
/*     fclose(ifile); */
/*   } */

/* sprintf(path,"%s%s.idb",buildpath,filestem); /\* SAVE THE ACTUAL INTERVAL DB*\/ */
/* ifile=fopen(path,"wb"); /\* binary file *\/ */
/* if (!ifile) */
/*   goto unable_to_open_file; */
/* for (i=0;i<npad;i++) { */
/*   if (NULL==fgets(line,32767,infile)) */
/*     goto fread_error_occurred; */
/*   if (6!=sscanf(line,"M %d %d %d %d %d %d",&(im.start),&(im.end), */
/* 	  &(im.target_id),&(im.target_start), */
/* 	  &(im.target_end),&(im.sublist))) */
/*     goto fread_error_occurred; */
/*   if (1!=fwrite(&im,sizeof(IntervalMap),1,ifile)) */
/*     goto write_error_occurred; */
/* } */
/* fclose(ifile); */

/*   return 0; /\* INDICATES NO ERROR OCCURRED *\/ */
/*  unable_to_open_file: */
/*   if (err_msg) */
/*     sprintf(err_msg,"unable to open file %s",path); */
/*   return -1; */
/*  fread_error_occurred: */
/*   if (err_msg) */
/*     sprintf(err_msg,"error or EOF reading input file"); */
/*   return -1; */
/*  write_error_occurred: */
/*   if (err_msg) */
/*     sprintf(err_msg,"error writing file %s! out of disk space?", */
/* 	    path); */
/*   return -1; */
/* } */



/* int main(int argc, char **argv) { */

/*   int interval_map_size = 1024; */

/*   IntervalMap *im; */
/*   int len = 10000000; */
/*   SublistHeader *sl; */
/*   int *p_n = malloc(sizeof *p_n); */
/*   int *p_nlists = malloc(sizeof *p_nlists); */
/*   int *nhits = malloc(sizeof *nhits); */

/*   FILE *ifp; */

/*   ifp = fopen("../test.csv", "r"); */

/*   struct timeval  tv1, tv2; */
/*   gettimeofday(&tv1, NULL); */
/*   im = read_intervals(len, ifp); */
/*   gettimeofday(&tv2, NULL); */
/*   printf ("Total time = %f seconds\n", */
/*           (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + */
/*           (double) (tv2.tv_sec - tv1.tv_sec)); */

/*   gettimeofday(&tv1, NULL); */
/*   sl = build_nested_list(im, len, p_n, p_nlists); */
/*   gettimeofday(&tv2, NULL); */
/*   printf ("Total time = %f seconds\n", */
/*           (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + */
/*           (double) (tv2.tv_sec - tv1.tv_sec)); */
/*   IntervalIterator *it = interval_iterator_alloc(); */

/*   IntervalMap im_buf[interval_map_size]; */

/*   printf("*p_nlists %d\n", *p_nlists); */
/*   find_intervals(it, 0, 500, im, len, sl, *p_nlists, im_buf, interval_map_size, nhits, &it); */
/*   printf("*nhits %d\n", *nhits); */

/*   int i; */
/*   for (i = 0; i < *nhits; i++){ */
/*     printf("Start %d End %d Id %d\n", im_buf[i].target_start, im_buf[i].target_end, im_buf[i].target_id); */
/*   } */

/*   free(p_n); */
/*   free(p_nlists); */
/*   free(nhits); */

/* } */


/* int find_k_next(int start, int end, */
/*                 IntervalMap im[], int n, */
/*                 SublistHeader subheader[], int nlists, */
/*                 IntervalMap buf[], int ktofind, */
/*                 int *p_nreturn) */
/* { */
/*   IntervalIterator *it=NULL,*it2=NULL; */
/*   int nfound=0,j,k; */
/*   /\* IntervalMap *results = interval_map_alloc(ktofind); *\/ */

/*   /\* CALLOC(it,1,IntervalIterator); *\/ */

/*   if (it->n == 0) { /\* DEFAULT: SEARCH THE TOP NESTED LIST *\/ */
/*     it->n=n; */
/*     it->i=find_overlap_start(start,end,im,n); */
/*   } */

/*   do { */
/*     while (it->i>=0 && it->i<it->n && (nfound < ktofind)) { */
/*       if (!HAS_OVERLAP_POSITIVE(im[it->i],start,end)) { */
/*         buf[nfound] = im[it->i]; /\*SAVE THIS HIT TO BUFFER *\/ */
/*         nfound++; */
/*       } */
/*       k=im[it->i].sublist; /\* GET SUBLIST OF i IF ANY *\/ */
/*       it->i++; /\* ADVANCE TO NEXT INTERVAL *\/ */
/*       if (k>=0 && (j=find_suboverlap_start(start,end,k,im,subheader))>=0) { */
/*         PUSH_ITERATOR_STACK(it,it2,IntervalIterator); /\* RECURSE TO SUBLIST *\/ */
/*         it2->i = j; /\* START OF OVERLAPPING HITS IN THIS SUBLIST *\/ */
/*         it2->n = subheader[k].start+subheader[k].len; /\* END OF SUBLIST *\/ */
/*         it=it2; /\* PUSH THE ITERATOR STACK *\/ */
/*       } */
/*     } */
/*   } while (POP_ITERATOR_STACK(it));  /\* IF STACK EXHAUSTED, EXIT *\/ */
/*   free_interval_iterator(it); /\* takes care of the whole stack *\/ */
/*   it=NULL;  /\* ITERATOR IS EXHAUSTED *\/ */

/*   *p_nreturn=nfound; /\* #INTERVALS FOUND IN THIS PASS *\/ */
/* } */


int find_intervals_stack(int start_stack[], int end_stack[], int sp, int start,
                                int end, IntervalMap im[], int n,
                                SublistHeader subheader[], IntervalMap buf[],
                                int *p_nreturn)
{
  /* IntervalIterator *it=NULL,*it2=NULL; */
  /* printf("In very beginning!\n"); */
  /* return 0; */
  int nfound = 0, j, k;
  /* printf("j: %d, sp: %d, start_stack[sp]: %d", 0, sp, sp); */

  /* if (sp == 0) { */
  clock_t t;
  t = clock();
  j = find_overlap_start(start,end,im,n);
  t = clock() - t;
  double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
  printf("fun() took %f seconds to execute \n", time_taken);
  start_stack[sp] = j;
  end_stack[sp] = n;
  /* } */

  /* printf("We are before loop\n"); */
  /* printf("start, end: %d, %d", start_stack[sp], end_stack[sp]); */

  /* fflush(stdout); */

  while (sp >= 0) {
    /* printf("Outer loop. sp: %d, st: %d, end: %d\n", sp, start_stack[sp], end_stack[sp]); */
    /* fflush(stdout); */
    while (start_stack[sp] >= 0 && start_stack[sp] < end_stack[sp] && \
           HAS_OVERLAP_POSITIVE(im[start_stack[sp]], start, end)) {
      /* printf("Inner loop. sp: %d\n", start_stack[sp]); */
      /* printf("Interval added: %d, %d, %d\n", im[start_stack[sp]].start, im[start_stack[sp]].end, im[start_stack[sp]].target_id); */
      memcpy(buf+nfound, im + start_stack[sp], sizeof(IntervalMap)); /*SAVE THIS HIT TO BUFFER */

      nfound++;
      k=im[sp].sublist; /* GET SUBLIST OF i IF ANY */

      start_stack[sp++]++; /* ADVANCE TO NEXT INTERVAL */
      if (k>=0 && (j=find_suboverlap_start(start,end,k,im,subheader))>=0) {
        sp++;
        start_stack[sp] = j;
        end_stack[sp] = subheader[k].start + subheader[k].len; /* END OF SUBLIST */
      }

      if (nfound>=1024){ /* FILLED THE BUFFER, RETURN THE RESULTS SO FAR */
        goto finally_return_result;
      }
    }

    sp--;

  }

 finally_return_result:

    *p_nreturn = nfound; /* #INTERVALS FOUND IN THIS PASS */

  return sp;
}

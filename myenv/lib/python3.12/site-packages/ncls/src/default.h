
#ifndef DEFAULT_HEADER_INCLUDED
#define DEFAULT_HEADER_INCLUDED 1

/* ON LINUX, GIVE US SUPPORT FOR LARGE FILES BY DEFAULT */
#define _FILE_OFFSET_BITS 64
#define PYGR_OFF_T off_t

/* try to get 64 bit fseek on Windows if available, otherwise fall
   back to regular fseek version.  On other platforms use POSIX fseeko */
#ifdef __MSVCRT__
#define PYGR_FSEEK(IFILE,OFFSET,WHENCE) fseeko64(IFILE,OFFSET,WHENCE)
#elif defined(_WIN32)
#define PYGR_FSEEK(IFILE,OFFSET,WHENCE) fseek(IFILE,OFFSET,WHENCE)
#else
#define PYGR_FSEEK(IFILE,OFFSET,WHENCE) fseeko(IFILE,OFFSET,WHENCE)
#endif

#ifdef BUILD_C_LIBRARY
#include <sys/types.h>
#else
#include "Python.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#if defined(__STDC__) || defined(__ANSI_CPP__)  /*###################*/
#define STRINGIFY(NAME) # NAME
#else
#define STRINGIFY(NAME) "NAME"
#endif

#define FREE(P) if (P) {free(P);(P)=NULL;}


#ifndef BUILD_C_LIBRARY
/* USE THESE DEFINITIONS FOR BUILDING A PYTHON EXTENSION MODULE  *****************************/


/* IF YOU USE CALLOC, YOUR FUNCTION MUST DEFINE A HANDLER WITH LABEL
   handle_malloc_failure:
   THIS HANDLER SHOULD RELEASE ANY TEMPORARILY ALLOCATED MEMORY AND
   return AN ERROR CODE SIGNALING ITS CALLER THAT IT WAS UNABLE TO
   ALLOCATE THE DESIRED RESOURCE...  */
#define MALLOC_FAILURE_ACTION goto handle_malloc_failure

/* USE OF THIS MACRO WILL HANDLE MALLOC FAILURES BY
   RAISING THE APPROPRIATE PYTHON EXCEPTIONS
   INSTEAD OF CRASHING PYTHON OR FORCING THE PROCESS TO ABORT... */
#define CALLOC(memptr,N,ATYPE) \
  if ((N)<=0) {\
    char errstr[1024]; \
    sprintf(errstr,"%s, line %d: *** invalid memory request: %s[%d].\n",\
              __FILE__,__LINE__,STRINGIFY(memptr),(N));   \
    PyErr_SetString(PyExc_ValueError,errstr); \
    MALLOC_FAILURE_ACTION;\
  }\
  else if (NULL == ((memptr)=(ATYPE *)calloc((size_t)(N),sizeof(ATYPE))))  { \
    char errstr[1024]; \
    sprintf(errstr,"%s, line %d: memory request failed: %s[%d].\n",\
              __FILE__,__LINE__,STRINGIFY(memptr),(N));   \
    PyErr_SetString(PyExc_MemoryError,errstr); \
    MALLOC_FAILURE_ACTION;\
  }

/* IF realloc FAILS, memptr REMAINS VALID, BUT MALLOC_FAILURE_ACTION IS INVOKED. */
#define REALLOC(memptr,N,ATYPE) \
  if ((N)<=0) {\
    char errstr[1024]; \
    sprintf(errstr,"%s, line %d: *** invalid memory request: %s[%d].\n",\
              __FILE__,__LINE__,STRINGIFY(memptr),(N));   \
    PyErr_SetString(PyExc_ValueError,errstr); \
    MALLOC_FAILURE_ACTION;\
  }\
  else {\
    void *tmp_realloc_ptrZZ; \
    if (NULL == (tmp_realloc_ptrZZ=realloc((memptr),(size_t)(N)*sizeof(ATYPE))))  { \
      char errstr[1024]; \
      sprintf(errstr,"%s, line %d: memory request failed: %s[%d].\n",\
                __FILE__,__LINE__,STRINGIFY(memptr),(N));   \
      PyErr_SetString(PyExc_MemoryError,errstr); \
      MALLOC_FAILURE_ACTION;\
    } \
    else \
      (memptr)=(ATYPE *)tmp_realloc_ptrZZ; \
  }


#else
/* USE THESE DEFINITIONS FOR BUILDING A C LIBRARY *****************************/
#define MALLOC_FAILURE_ACTION abort()
#define CALLOC(memptr,N,ATYPE) \
  if ((N)<=0) {\
    fprintf(stderr,"%s, line %d: *** invalid memory request: %s[%d].\n",\
              __FILE__,__LINE__,STRINGIFY(memptr),(N));   \
    MALLOC_FAILURE_ACTION;\
  }\
  else if (NULL == ((memptr)=(ATYPE *)calloc((size_t)(N),sizeof(ATYPE))))  { \
    fprintf(stderr,"%s, line %d: memory request failed: %s[%d].\n",\
              __FILE__,__LINE__,STRINGIFY(memptr),(N));   \
    MALLOC_FAILURE_ACTION;\
  }

/* IF realloc FAILS, memptr REMAINS VALID, BUT MALLOC_FAILURE_ACTION IS INVOKED. */
#define REALLOC(memptr,N,ATYPE) \
  if ((N)<=0) {\
    fprintf(stderr,"%s, line %d: *** invalid memory request: %s[%d].\n",\
              __FILE__,__LINE__,STRINGIFY(memptr),(N));   \
    MALLOC_FAILURE_ACTION;\
  }\
  else {\
    void *tmp_realloc_ptrZZ; \
    if (NULL == (tmp_realloc_ptrZZ=realloc((memptr),(size_t)(N)*sizeof(ATYPE))))  { \
      fprintf(stderr,"%s, line %d: memory request failed: %s[%d].\n",\
                __FILE__,__LINE__,STRINGIFY(memptr),(N));   \
      MALLOC_FAILURE_ACTION;\
    } \
    else \
      (memptr)=(ATYPE *)tmp_realloc_ptrZZ; \
  }
#endif



#endif

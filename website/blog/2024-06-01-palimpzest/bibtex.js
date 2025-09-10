import React, { useState } from "react"


const BibtexInner = ({children, handleCopyClick}) => (
  <div
    className="bibtex"
    style={{
      position: "relative",
      backgroundColor: "#f5f5f5",
      padding: "15px",
      borderRadius: "3px",
      margin: "15px 0"
    }}
  >
    <pre style={{margin: 0, fontSize: "0.9em"}}>
      {children}
    </pre>
    <button
      className="bibtex-copy"
      onClick={handleCopyClick}
      style={{
        position: "absolute",
        top: "10px",
        right: "10px",
        cursor: "pointer",
        backgroundColor: "#aaa",
        color: "#fff",
        border: "0",
        borderRadius: "3px",
        padding: "5px",
        outline: "none",
        fontSize: "0.8em",
        fontFamily: "sans-serif",
      }}
    >
      Copy
    </button>
  </div>
)


const Bibtex = ({children, withToggle}) => {
  const [visible, setVisible] = useState(false);

  const handleToggle = (event) => {
    setVisible(!visible)
    event.preventDefault()
  }

  const handleCopyClick = () => {
    navigator.clipboard.writeText(children)
  }

  if (withToggle) {
    if (!visible) {
      return (
        <>
          <a href="#" onClick={handleToggle}>View BibTeX</a>
        </>
      )
    } else {
        return (
          <>
            <a href="#" onClick={handleToggle}>Hide BibTeX</a>
            <BibtexInner handleCopyClick={handleCopyClick}>{children}</BibtexInner>
          </>
        )
    }
  } else {
    return <BibtexInner handleCopyClick={handleCopyClick}>{children}</BibtexInner>
  }
}

export default Bibtex

import "./App.css";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import React from "react";
import { StyleSheet, View } from "react-native-web";

function App() {
  const [value, setValue] = React.useState("");
  const [translated, setTranslated] = React.useState("");

  const handleChange = (event) => {
    setValue(event.target.value);
  };

  const buttonClick = (event) => {
    console.log("button clicked!");
    setTranslated("loading...");

    fetch("/server/translate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: value,
      }),
    })
      .then((res) => res.text())
      .then((translated) => {
        setTranslated(translated);
        console.log(translated);
      });

    // alert("작성 완료");
  };

  const clearClick = (event) => {
    console.log("clear clicked!");

    setValue("");
    setTranslated("");
  };

  const styles = StyleSheet.create({
    container: {
      flex: 1,
      padding: 20,
    },
  });

  return (
    <div>
      <div>
        <div
          className="app-header"
          style={{
            textAlign: "center",
          }}
        >
          <h2 className="header">C Code Expression Generator</h2>
        </div>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >

        <TextField
          id="outlined-multiline-static"
          label="Texts"
          multiline
          rows={15}
          value={value}
          onChange={handleChange}
          style={{
            width: "100%",
          }}
        />

        <View
          style={[
            styles.containor,
            {
              flexDirection: "column",
            },
          ]}
        >
          <Button
            variant="contained"
            onClick={buttonClick}
            style={{
              margin: "20px",
            }}
          >
            generate 
          </Button>

          <Button
            variant="contained"
            onClick={clearClick}
            style={{
              margin: "20px",
            }}
          >
            - delete -
          </Button>
        </View>

        <TextField
          id="outlined-multiline-static"
          // label="Line Breaked Texts"
          multiline
          rows={15}
          defaultValue={translated}
          inputProps={{
            readOnly: true,
          }}
          style={{
            width: "100%",
          }}
        />
      </div>
    </div>
  );
}

export default App;
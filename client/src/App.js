import "./App.css";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import React from "react";
import { StyleSheet, View, Text, TouchableOpacity,} from "react-native-web";
// import { StyleSheet, Text, TouchableOpacity, View } from "react-native";

function App() {


  const [value, setValue] = React.useState("");
  const [translated, setTranslated] = React.useState({});


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
      .then((res) => res.json())
      .then((translated) => {
        setTranslated(translated);
        console.log(translated);
      });
  };


  const clearClick = (event) => {
    console.log("clear clicked!");

    setValue("");
    setTranslated({});
  };

  const styles = StyleSheet.create({
    container: {
      flex: 1,
      padding: 20,
    },
  });

  return (
    <View style={[styles.container, {
      // Try setting `flexDirection` to `"row"`.
      flexDirection: "column",
      alignItems: "center"
    }]}>

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


      <TextField
        id="outlined-multiline-static"
        label="Source Code"
        multiline
        rows={15}
        value={value}
        onChange={handleChange}
        style={{
          margin: "1%",
          width: "70%"
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
        helperText="Patch #2"
        rows={1}
        multiline
        defaultValue={translated.patch_1}
        inputProps={{
          readOnly: true,
        }}
        style={{
          marginTop: "30px",
          margin: "1%",
          width: "70%"
        }}
      />

      <TextField
        id="outlined-multiline-static"
        helperText="Patch #2"
        rows={1}
        multiline
        defaultValue={translated.patch_2}
        inputProps={{
          readOnly: true,
        }}
        style={{
          marginTop: "30px",
          margin: "1%",
          width: "70%"
        }}
      />

      <TextField
        id="outlined-multiline-static"
        helperText="Patch #3"
        rows={1}
        multiline
        defaultValue={translated.patch_3}
        InputProps={{
          readOnly: true,
        }}
        style={{
          marginTop: "30px",
          margin: "1%",
          width: "70%"
        }}
      />

      <TextField
        id="outlined-multiline-static"
        helperText="Patch #4"
        rows={1}
        multiline
        defaultValue={translated.patch_4}
        inputProps={{
          readOnly: true,
        }}
        style={{
          marginTop: "30px",
          margin: "1%",
          width: "70%"
        }}
      />

      <TextField
        id="outlined-multiline-static"
        helperText="Patch #5"
        rows={1}
        multiline
        defaultValue={translated.patch_5}
        inputProps={{
          readOnly: true,
        }}
        style={{
          marginTop: "30px",
          margin: "1%",
          width: "70%"
        }}
      />
      
    </View>
  );
}

export default App;
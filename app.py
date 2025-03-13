from flask import Flask, render_template, request, session
from FinalModel import CGLPredictor

app = Flask(__name__)
app.secret_key = '?87Rc:"m4C_]R>b8hL;f+nEFZ=o$6r!gcU'  # Required for using session
predictor = CGLPredictor()

@app.route('/')
def home():
    # Render input.html with session-stored values or default empty values
    return render_template('input.html',
                           width=session.get('width', ""),
                           thickness=session.get('thickness', ""),
                           gsm_a=session.get('gsm_a', ""),
                           hardness=session.get('hardness', ""),
                           error="")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and validate form data
        width = float(request.form['width'])
        thickness = float(request.form['thickness'])
        gsm_a = float(request.form['gsm_a'])
        hardness = float(request.form['hardness'])
        features = [width, thickness, gsm_a, hardness]

        # Store the inputs in session
        session['width'] = width
        session['thickness'] = thickness
        session['gsm_a'] = gsm_a
        session['hardness'] = hardness

    except ValueError:
        # If validation fails, return to input page with error message
        return render_template('input.html',
                               width=request.form.get('width', ""),
                               thickness=request.form.get('thickness', ""),
                               gsm_a=request.form.get('gsm_a', ""),
                               hardness=request.form.get('hardness', ""),
                               error="Error: All inputs must be valid numbers.")

    # Main prediction
    main_output = predictor.predict_main(features)
    
    # Extract values for TPH calculation
    speed = main_output['Speed']
    
    # Corrected TPH formula with density (7850 kg/m³)
    tph = (60 * width * thickness * speed * 7850) / 1e9
    
    # Round all outputs
    rounded_main = {k: round(v) for k, v in main_output.items()}
    rounded_firing = round(predictor.predict_firing(
        [main_output[f'NOF{i}'] for i in range(1, 6)],
        speed
    ))
    
    return render_template('output.html',
                           main=rounded_main,
                           firing=rounded_firing,
                           tph=round(tph))

if __name__ == '__main__':
    app.run(debug=True)

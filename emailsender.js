// Enable GMAIL API to send email through cloud function using below code

const nodemailer = require("nodemailer");

const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user: 'YOUR_GMAIL_ID',
        pass: 'YOUR_GMAIL_PASSWORD'
    }
});

const mailOptions = {
    from: "FROM_NAME", // sender address
    to: "TO_EMAIL", // list of receivers
    subject: "EMAIL_SUBJECT", // Subject line
    html: "<p> EMAIL_HTML_BODY </p>"
};

transporter.sendMail(mailOptions, function (err, info) {
    if(err)
    {
      console.log(err);
    }
});
